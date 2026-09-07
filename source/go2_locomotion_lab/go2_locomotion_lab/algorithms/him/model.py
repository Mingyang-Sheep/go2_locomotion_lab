"""RSL-RL v5 model implementing the HIM estimator-to-actor data flow."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .estimator import HimEstimator


class HimActorModel(MLPModel):
    """Actor consuming proprio history through a detached HIM estimator."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_steps: int = 6,
        one_step_obs_dim: int = 45,
        latent_dim: int = 16,
        estimator_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        estimator_target_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        estimator_num_prototypes: int = 32,
        estimator_learning_rate: float = 1.0e-3,
        **kwargs,
    ) -> None:
        self.history_steps = history_steps
        self.one_step_obs_dim = one_step_obs_dim
        self.him_latent_dim = latent_dim
        if kwargs.get("obs_normalization", False):
            raise ValueError("HIM history normalization must be implemented explicitly; set obs_normalization=False")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != history_steps * one_step_obs_dim:
            raise ValueError(
                f"HIM actor expects {history_steps}x{one_step_obs_dim}={history_steps * one_step_obs_dim} "
                f"observations, got {self.obs_dim}"
            )
        self.estimator = HimEstimator(
            history_steps=history_steps,
            one_step_obs_dim=one_step_obs_dim,
            hidden_dims=estimator_hidden_dims,
            target_hidden_dims=estimator_target_hidden_dims,
            latent_dim=latent_dim,
            num_prototypes=estimator_num_prototypes,
            learning_rate=estimator_learning_rate,
        )

    def _get_latent_dim(self) -> int:
        return self.one_step_obs_dim + 3 + self.him_latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        history = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        velocity, embedding = self.estimator(history)
        current = history[..., -self.one_step_obs_dim :]
        return torch.cat((current, velocity, embedding), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        return _HimPolicyExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _HimPolicyOnnxExport(self, verbose)


class _HimPolicyExport(nn.Module):
    def __init__(self, model: HimActorModel) -> None:
        super().__init__()
        self.one_step_obs_dim = model.one_step_obs_dim
        self.encoder = copy.deepcopy(model.estimator.encoder)
        self.actor = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        estimate = self.encoder(history)
        velocity = estimate[..., :3]
        embedding = torch.nn.functional.normalize(estimate[..., 3:], dim=-1)
        latent = torch.cat((history[..., -self.one_step_obs_dim :], velocity, embedding), dim=-1)
        return self.deterministic_output(self.actor(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _HimPolicyOnnxExport(_HimPolicyExport):
    def __init__(self, model: HimActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["proprio_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
