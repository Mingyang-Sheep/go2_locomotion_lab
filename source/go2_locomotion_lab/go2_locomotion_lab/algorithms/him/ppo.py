"""HIM estimator optimization layered onto official RSL-RL PPO."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from .model import HimActorModel


class HimPPO(PPO):
    """Keep PPO behavior upstream while training the HIM estimator independently."""

    actor: HimActorModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        critic_shape = self.storage.observations["critic"].shape
        self.next_critic_observations = torch.zeros(critic_shape, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.next_critic_observations[self.storage.step].copy_(obs["critic"])
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        history = self.storage.observations["policy"].flatten(0, 1)
        next_privileged = self.next_critic_observations.flatten(0, 1)
        non_terminal = ~self.storage.dones.flatten(0, 1).squeeze(-1).bool()
        valid_indices = non_terminal.nonzero(as_tuple=False).squeeze(-1)
        mini_batch_count = max(1, self.num_mini_batches)
        velocity_loss = 0.0
        swap_loss = 0.0
        update_count = 0
        for _ in range(self.num_learning_epochs):
            permutation = valid_indices[torch.randperm(valid_indices.numel(), device=history.device)]
            for indices in permutation.tensor_split(mini_batch_count):
                if indices.numel() == 0:
                    continue
                vel, swap = self.actor.estimator.update(
                    history[indices],
                    next_privileged[indices, : self.actor.one_step_obs_dim],
                    next_privileged[indices, self.actor.one_step_obs_dim : self.actor.one_step_obs_dim + 3],
                )
                velocity_loss += vel
                swap_loss += swap
                update_count += 1

        losses = super().update()
        losses["him_velocity"] = velocity_loss / update_count if update_count else 0.0
        losses["him_swap"] = swap_loss / update_count if update_count else 0.0
        return losses

    def save(self) -> dict:
        state = super().save()
        state["estimator_optimizer_state_dict"] = self.actor.estimator.optimizer.state_dict()
        return state

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", True):
            estimator_state = loaded_dict.get("estimator_optimizer_state_dict")
            if estimator_state is not None:
                self.actor.estimator.optimizer.load_state_dict(estimator_state)
        return load_iteration
