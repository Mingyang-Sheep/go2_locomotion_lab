"""Hybrid internal model estimator adapted from the HIMLoco formulation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


def _activation(name: str) -> nn.Module:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported HIM activation: {name}") from exc


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, iterations: int = 3) -> torch.Tensor:
    """Compute balanced soft prototype assignments."""

    assignments = torch.exp(scores / epsilon).T
    assignments /= assignments.sum().clamp_min(1.0e-12)
    prototypes, batch = assignments.shape
    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        assignments /= prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        assignments /= batch
    return (assignments * batch).T


class HimEstimator(nn.Module):
    """Estimate body velocity and a contrastive terrain/dynamics embedding."""

    def __init__(
        self,
        history_steps: int,
        one_step_obs_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        target_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        latent_dim: int = 16,
        num_prototypes: int = 32,
        activation: str = "elu",
        temperature: float = 3.0,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 10.0,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.one_step_obs_dim = one_step_obs_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm

        encoder_layers: list[nn.Module] = []
        input_dim = history_steps * one_step_obs_dim
        for width in hidden_dims:
            encoder_layers.extend((nn.Linear(input_dim, width), _activation(activation)))
            input_dim = width
        encoder_layers.append(nn.Linear(input_dim, 3 + latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        target_layers: list[nn.Module] = []
        input_dim = one_step_obs_dim
        for width in target_hidden_dims:
            target_layers.extend((nn.Linear(input_dim, width), _activation(activation)))
            input_dim = width
        target_layers.append(nn.Linear(input_dim, latent_dim))
        self.target = nn.Sequential(*target_layers)
        self.prototypes = nn.Embedding(num_prototypes, latent_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(history)
        velocity = encoded[..., :3]
        latent = functional.normalize(encoded[..., 3:], dim=-1)
        return velocity.detach(), latent.detach()

    def update(
        self, history: torch.Tensor, current_proprio: torch.Tensor, target_velocity: torch.Tensor
    ) -> tuple[float, float]:
        encoded = self.encoder(history.detach())
        predicted_velocity = encoded[..., :3]
        source_latent = functional.normalize(encoded[..., 3:], dim=-1)
        target_latent = functional.normalize(self.target(current_proprio.detach()), dim=-1)

        with torch.no_grad():
            self.prototypes.weight.copy_(functional.normalize(self.prototypes.weight, dim=-1))
        source_scores = source_latent @ self.prototypes.weight.T
        target_scores = target_latent @ self.prototypes.weight.T
        with torch.no_grad():
            source_assignments = sinkhorn(source_scores)
            target_assignments = sinkhorn(target_scores)
        source_log_prob = functional.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_prob = functional.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (
            (source_assignments * target_log_prob).sum(dim=-1).mean()
            + (target_assignments * source_log_prob).sum(dim=-1).mean()
        )
        velocity_loss = functional.mse_loss(predicted_velocity, target_velocity.detach())
        loss = velocity_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return velocity_loss.item(), swap_loss.item()
