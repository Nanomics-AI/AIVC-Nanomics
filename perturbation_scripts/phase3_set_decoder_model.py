"""Frozen Phase-III D1/D2 set-level Top20 delta decoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


SET_SIZE = 256
LATENT_DIM = 768
HIDDEN_DIM = 256
GENE_DIM = 20


def _transformer_encoder() -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=HIDDEN_DIM,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=2)


class D1CNNTransformerEncoder(nn.Module):
    """Order-sensitive frozen CNN route; one instance is shared by both sets."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2)),
            nn.GELU(),
        )
        self.token_projection = nn.Linear(32, HIDDEN_DIM)
        self.transformer = _transformer_encoder()
        self.last_shapes: dict[str, list[int]] = {}

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 3 or tuple(cells.shape[1:]) != (SET_SIZE, LATENT_DIM):
            raise ValueError(f"D1 expected [B,{SET_SIZE},{LATENT_DIM}], got {tuple(cells.shape)}")
        x0 = cells.unsqueeze(1)
        x1 = self.convs[1](self.convs[0](x0))
        x2 = self.convs[3](self.convs[2](x1))
        x3 = self.convs[5](self.convs[4](x2))
        if tuple(x3.shape[1:]) != (32, 32, 12):
            raise AssertionError(f"Frozen D1 CNN shape changed: {tuple(x3.shape)}")
        tokens32 = x3.permute(0, 2, 3, 1).contiguous().reshape(len(cells), 384, 32)
        tokens256 = self.token_projection(tokens32)
        encoded = self.transformer(tokens256)
        self.last_shapes = {
            "input": list(cells.shape),
            "unsqueezed": list(x0.shape),
            "conv1": list(x1.shape),
            "conv2": list(x2.shape),
            "conv3": list(x3.shape),
            "tokens32": list(tokens32.shape),
            "tokens256": list(tokens256.shape),
            "transformer": list(encoded.shape),
            "pooled": [len(cells), HIDDEN_DIM],
        }
        return encoded.mean(dim=1)


class D2SetTransformerEncoder(nn.Module):
    """Permutation-invariant frozen Set Transformer route."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.transformer = _transformer_encoder()
        self.attention_pool = nn.Linear(HIDDEN_DIM, 1)
        self.last_shapes: dict[str, list[int]] = {}

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 3 or tuple(cells.shape[1:]) != (SET_SIZE, LATENT_DIM):
            raise ValueError(f"D2 expected [B,{SET_SIZE},{LATENT_DIM}], got {tuple(cells.shape)}")
        projected = self.input_projection(cells)
        encoded = self.transformer(projected)
        scores = self.attention_pool(encoded)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * encoded).sum(dim=1)
        self.last_shapes = {
            "input": list(cells.shape),
            "projected": list(projected.shape),
            "transformer": list(encoded.shape),
            "attention_scores": list(scores.shape),
            "attention_weights": list(weights.shape),
            "pooled": list(pooled.shape),
        }
        return pooled


class SetDeltaDecoder(nn.Module):
    """Shared set encoder followed by the frozen signed 20-gene delta readout."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        variant = variant.lower()
        if variant == "d1":
            self.shared_encoder: nn.Module = D1CNNTransformerEncoder()
        elif variant == "d2":
            self.shared_encoder = D2SetTransformerEncoder()
        else:
            raise ValueError("variant must be 'd1' or 'd2'")
        self.variant = variant
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 128),
            nn.GELU(),
            nn.Linear(128, GENE_DIM),
        )

    def forward(self, control: torch.Tensor, treated: torch.Tensor) -> torch.Tensor:
        control_hidden = self.shared_encoder(control)
        treated_hidden = self.shared_encoder(treated)
        latent_delta = treated_hidden - control_hidden
        return self.readout(latent_delta)


def build_set_decoder(variant: str, seed: int = 42) -> SetDeltaDecoder:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return SetDeltaDecoder(variant)


def architecture_audit(model: SetDeltaDecoder) -> dict[str, Any]:
    encoder = model.shared_encoder
    common = {
        "single_shared_encoder": not hasattr(model, "control_encoder")
        and not hasattr(model, "treated_encoder"),
        "readout_exact": isinstance(model.readout, nn.Sequential)
        and [type(layer).__name__ for layer in model.readout]
        == ["Linear", "GELU", "Linear", "GELU", "Linear"]
        and (model.readout[0].in_features, model.readout[0].out_features) == (256, 256)
        and (model.readout[2].in_features, model.readout[2].out_features) == (256, 128)
        and (model.readout[4].in_features, model.readout[4].out_features) == (128, 20),
        "signed_identity_output": isinstance(model.readout[-1], nn.Linear),
        "no_dropout": not any(isinstance(module, nn.Dropout) and module.p != 0 for module in model.modules()),
        "no_positional_encoding": not any("position" in name.lower() for name, _ in model.named_parameters()),
    }
    if model.variant == "d1":
        variant_checks = {
            "encoder_class": isinstance(encoder, D1CNNTransformerEncoder),
            "conv_contract": isinstance(encoder, D1CNNTransformerEncoder)
            and [(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding)
                 for m in encoder.convs if isinstance(m, nn.Conv2d)]
            == [
                (1, 8, (3, 5), (2, 4), (1, 2)),
                (8, 16, (3, 5), (2, 4), (1, 2)),
                (16, 32, (3, 5), (2, 4), (1, 2)),
            ],
            "token_projection_32_to_256": isinstance(encoder, D1CNNTransformerEncoder)
            and (encoder.token_projection.in_features, encoder.token_projection.out_features)
            == (32, 256),
            "mean_pooling": True,
        }
    else:
        variant_checks = {
            "encoder_class": isinstance(encoder, D2SetTransformerEncoder),
            "input_projection_768_to_256": isinstance(encoder, D2SetTransformerEncoder)
            and (encoder.input_projection.in_features, encoder.input_projection.out_features)
            == (768, 256),
            "attention_pool_256_to_1": isinstance(encoder, D2SetTransformerEncoder)
            and (encoder.attention_pool.in_features, encoder.attention_pool.out_features)
            == (256, 1),
            "permutation_invariant_design": True,
        }
    transformer = encoder.transformer
    layers = list(transformer.layers)
    transformer_checks = {
        "transformer_layers_2": len(layers) == 2,
        "transformer_contract": all(
            layer.self_attn.embed_dim == 256
            and layer.self_attn.num_heads == 8
            and layer.linear1.in_features == 256
            and layer.linear1.out_features == 1024
            and layer.linear2.in_features == 1024
            and layer.linear2.out_features == 256
            and layer.norm_first
            and layer.dropout.p == 0.0
            and layer.dropout1.p == 0.0
            and layer.dropout2.p == 0.0
            for layer in layers
        ),
    }
    checks = {**common, **variant_checks, **transformer_checks}
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "variant": model.variant,
        "checks": checks,
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "output_shape": ["B", GENE_DIM],
        "final_activation": "identity",
    }


__all__ = [
    "GENE_DIM",
    "HIDDEN_DIM",
    "LATENT_DIM",
    "SET_SIZE",
    "D1CNNTransformerEncoder",
    "D2SetTransformerEncoder",
    "SetDeltaDecoder",
    "architecture_audit",
    "build_set_decoder",
]
