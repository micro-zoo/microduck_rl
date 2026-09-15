"""Compatibility runner for adding closed-loop tracking feedback to mimic actors."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

_JOINT_COMMAND_DIM = 28
_OLD_ACTOR_OBS_DIM = 79
_FEEDBACK_ACTOR_OBS_DIM = 85


def _expand_feature_axis(
    value: torch.Tensor, *, fill_value: float
) -> torch.Tensor:
    """Insert root-position and linear-velocity features into an actor vector."""
    if value.shape[-1] != _OLD_ACTOR_OBS_DIM:
        return value

    expanded = torch.full(
        (*value.shape[:-1], _FEEDBACK_ACTOR_OBS_DIM),
        fill_value,
        dtype=value.dtype,
        device=value.device,
    )
    expanded[..., :_JOINT_COMMAND_DIM] = value[..., :_JOINT_COMMAND_DIM]
    # The old actor has target orientation immediately after its 28-D command.
    expanded[..., 31:37] = value[..., 28:34]
    # Its IMU angular velocity and proprioception then follow.
    expanded[..., 40:] = value[..., 34:]
    return expanded


def expand_actor_tracking_feedback_checkpoint(checkpoint: dict) -> bool:
    """Migrate a 79-D no-state-estimation actor to the 85-D feedback layout.

    The new feedback terms are the body-frame target-root displacement and base
    linear velocity.  Their first-layer weights are zero, so migration preserves
    the existing policy exactly until PPO learns to use the added feedback.
    """
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        return False

    first_layer = actor_state.get("mlp.0.weight")
    if not isinstance(first_layer, torch.Tensor):
        return False
    if first_layer.shape[-1] == _FEEDBACK_ACTOR_OBS_DIM:
        return False
    if first_layer.shape[-1] != _OLD_ACTOR_OBS_DIM:
        raise ValueError(
            "Cannot migrate actor observations: expected 79 or 85 inputs, got "
            f"{first_layer.shape[-1]}."
        )

    expanded_layer = torch.zeros(
        first_layer.shape[0],
        _FEEDBACK_ACTOR_OBS_DIM,
        dtype=first_layer.dtype,
        device=first_layer.device,
    )
    expanded_layer[:, :_JOINT_COMMAND_DIM] = first_layer[:, :_JOINT_COMMAND_DIM]
    expanded_layer[:, 31:37] = first_layer[:, 28:34]
    expanded_layer[:, 40:] = first_layer[:, 34:]
    actor_state["mlp.0.weight"] = expanded_layer

    for name, fill_value in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        value = actor_state.get(name)
        if isinstance(value, torch.Tensor):
            actor_state[name] = _expand_feature_axis(value, fill_value=fill_value)
    return True


class TrackingFeedbackMotionTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
    """Load legacy mimic policies after adding actor-side root-state feedback."""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not expand_actor_tracking_feedback_checkpoint(checkpoint):
            return super().load(
                path, load_cfg=load_cfg, strict=strict, map_location=map_location
            )

        # Adam moments for the old first layer have the obsolete shape.  The
        # policy and critic still transfer exactly; start a fresh optimizer only
        # for this one architecture migration.
        effective_load_cfg = dict(load_cfg or {})
        if load_cfg is None:
            effective_load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        else:
            effective_load_cfg["optimizer"] = False

        load_iteration = self.alg.load(checkpoint, effective_load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = checkpoint["iter"]

        infos = checkpoint.get("infos")
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"][
                "common_step_counter"
            ]
        print("[INFO] Added actor root-state feedback; transferred policy weights.")
        return infos or {}
