import torch

from mjlab_microduck.tasks.tracking_feedback_runner import (
    expand_actor_tracking_feedback_checkpoint,
)


def test_feedback_checkpoint_expansion_preserves_old_policy_features():
    old_weight = torch.arange(4 * 79, dtype=torch.float32).reshape(4, 79)
    old_mean = torch.arange(79, dtype=torch.float32).reshape(1, 79)
    checkpoint = {
        "actor_state_dict": {
            "mlp.0.weight": old_weight.clone(),
            "obs_normalizer._mean": old_mean.clone(),
            "obs_normalizer._var": old_mean.clone() + 2.0,
            "obs_normalizer._std": old_mean.clone() + 3.0,
        }
    }

    assert expand_actor_tracking_feedback_checkpoint(checkpoint)
    state = checkpoint["actor_state_dict"]
    weight = state["mlp.0.weight"]
    assert weight.shape == (4, 85)
    torch.testing.assert_close(weight[:, :28], old_weight[:, :28])
    torch.testing.assert_close(weight[:, 31:37], old_weight[:, 28:34])
    torch.testing.assert_close(weight[:, 40:], old_weight[:, 34:])
    torch.testing.assert_close(weight[:, 28:31], torch.zeros(4, 3))
    torch.testing.assert_close(weight[:, 37:40], torch.zeros(4, 3))

    mean = state["obs_normalizer._mean"]
    variance = state["obs_normalizer._var"]
    standard_deviation = state["obs_normalizer._std"]
    assert mean.shape == (1, 85)
    torch.testing.assert_close(mean[:, 28:31], torch.zeros(1, 3))
    torch.testing.assert_close(mean[:, 37:40], torch.zeros(1, 3))
    torch.testing.assert_close(variance[:, 28:31], torch.ones(1, 3))
    torch.testing.assert_close(standard_deviation[:, 37:40], torch.ones(1, 3))
