"""Minimal Phase-II ST-A v2 conditioning wrapper around official STATE."""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import torch
import torch.nn as nn


LATENT_DIM = 768
DRUG_COUNT = 379
SET_SIZE = 256
EXPECTED_TOTAL_PARAMETERS = 101_558_784
EXPECTED_TRAINABLE_PARAMETERS = 76_982_784


def model_kwargs() -> dict[str, Any]:
    return {
        "input_dim": LATENT_DIM,
        "hidden_dim": LATENT_DIM,
        "output_dim": LATENT_DIM,
        # The official constructor requires pert_dim, but this temporary branch is deleted below.
        "pert_dim": 0,
        "predict_residual": False,
        "residual_mode": "output",
        "final_activation": "identity",
        "distributional_loss": "energy",
        "transformer_backbone_key": "llama",
        "transformer_backbone_kwargs": {
            "bidirectional_attention": True,
            "max_position_embeddings": SET_SIZE,
            "hidden_size": LATENT_DIM,
            "intermediate_size": 3072,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "use_cache": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "rotary_dim": 0,
            "use_rotary_embeddings": False,
        },
        "output_space": "embedding",
        "embed_key": "X_author_genejepa_epoch49",
        "gene_decoder_bool": False,
        "cell_set_len": SET_SIZE,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "activation": "gelu",
        "dropout": 0.0,
        "loss": "energy",
        "blur": 0.05,
        "mmd_num_chunks": 1,
        "randomize_mmd_chunks": False,
        "extra_tokens": 0,
        "batch_encoder": False,
        "batch_predictor": False,
        "use_batch_token": False,
        "confidence_token": False,
        "finetune_vci_decoder": False,
        "log1p_from_raw_counts": False,
        "lora": {"enable": False},
    }


def build_stav2(seed: int = 42) -> tuple[nn.Module, dict[str, Any]]:
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    class Phase2STAv2(StateTransitionPerturbationModel):
        """Official STATE body with only the frozen Phase-II conditioning replaced."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            del self.pert_encoder
            self.drug_embedding = nn.Embedding(DRUG_COUNT, LATENT_DIM)

        def conditioning(
            self,
            control: torch.Tensor,
            drug_id: torch.Tensor,
            dose_scaled: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if control.ndim != 3 or control.shape[1:] != (SET_SIZE, LATENT_DIM):
                raise ValueError(f"Unexpected control shape: {tuple(control.shape)}")
            drug_id = drug_id.reshape(-1).long()
            dose_scaled = dose_scaled.reshape(-1, 1).to(dtype=control.dtype)
            if len(drug_id) != len(control) or len(dose_scaled) != len(control):
                raise ValueError("Conditioning batch dimensions disagree")
            drug = self.drug_embedding(drug_id)
            drug_dose = drug * dose_scaled
            conditioned = control + drug_dose.unsqueeze(1)
            return conditioned, drug, drug_dose

        def forward(self, batch: dict[str, torch.Tensor], padded: bool = True) -> torch.Tensor:
            if not padded:
                raise ValueError("Phase-II formal ST-A v2 requires padded S=256 sets")
            basal = batch["ctrl_cell_emb"].reshape(-1, SET_SIZE, LATENT_DIM)
            conditioned, _, _ = self.conditioning(
                basal, batch["drug_id"], batch["dose_scaled"]
            )
            sequence = self.encode_basal_expression(conditioned)
            hidden = self.transformer_backbone(inputs_embeds=sequence).last_hidden_state
            self._token_features = hidden
            self._batch_token_cache = None
            output = self.project_out(hidden)
            if self.apply_output_relu:
                output = self.relu(output)
            return output.reshape(-1, self.output_dim)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs = model_kwargs()
    with contextlib.redirect_stdout(io.StringIO()):
        model = Phase2STAv2(**kwargs)
    return model, kwargs


def architecture_audit(model: nn.Module) -> dict[str, Any]:
    from geomloss import SamplesLoss

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_keys = list(model.state_dict())
    config = model.transformer_backbone.config
    checks = {
        "drug_embedding_shape": tuple(model.drug_embedding.weight.shape) == (DRUG_COUNT, LATENT_DIM),
        "drug_embedding_trainable": model.drug_embedding.weight.requires_grad,
        "pert_encoder_absent": not hasattr(model, "pert_encoder")
        and not any(key.startswith("pert_encoder.") for key in state_keys),
        "basal_encoder_linear_768": isinstance(model.basal_encoder, nn.Sequential)
        and len(model.basal_encoder) == 1
        and isinstance(model.basal_encoder[0], nn.Linear)
        and (model.basal_encoder[0].in_features, model.basal_encoder[0].out_features)
        == (LATENT_DIM, LATENT_DIM),
        "project_out_linear_768": isinstance(model.project_out, nn.Sequential)
        and len(model.project_out) == 1
        and isinstance(model.project_out[0], nn.Linear)
        and (model.project_out[0].in_features, model.project_out[0].out_features)
        == (LATENT_DIM, LATENT_DIM),
        "llama_bidirectional": type(model.transformer_backbone).__name__
        == "LlamaBidirectionalModel",
        "eight_layers": config.num_hidden_layers == 8,
        "twelve_heads": config.num_attention_heads == 12,
        "hidden_768": config.hidden_size == LATENT_DIM,
        "intermediate_3072": config.intermediate_size == 3072,
        "no_rotary": config.use_rotary_embeddings is False,
        "absolute_prediction": model.predict_residual is False,
        "identity_output": model.final_activation_name == "identity"
        and model.apply_output_relu is False,
        "energy_blur_005": isinstance(model.loss_fn, SamplesLoss)
        and model.loss_fn.loss == "energy"
        and model.loss_fn.blur == 0.05,
        "disabled_extra_heads": model.batch_encoder is None
        and not model.batch_predictor
        and not model.use_batch_token
        and model.confidence_token is None
        and model.gene_decoder is None,
        "total_parameters_exact": total == EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters_exact": trainable == EXPECTED_TRAINABLE_PARAMETERS,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "drug_embedding_parameters": model.drug_embedding.weight.numel(),
        "state_dict_has_drug_embedding": "drug_embedding.weight" in state_keys,
        "state_dict_pert_encoder_keys": [
            key for key in state_keys if key.startswith("pert_encoder.")
        ],
    }


if __name__ == "__main__":
    instance, _ = build_stav2()
    result = architecture_audit(instance)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)
