import pytest
import torch
from torch.optim import SGD

from stable_ssl.schedulers import (
    CosineDecayer,
    LinearWarmup,
    LinearWarmupCosineAnnealing,
    LinearWarmupCyclicAnnealing,
    LinearWarmupThreeStepsAnnealing,
)


@pytest.fixture
def dummy_model_optimizer():
    """Create a dummy model and an SGD optimizer with an initial LR of 0.1."""
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    return model, optimizer


def current_lr(optimizer):
    """Get the current learning rate from an optimizer."""
    return optimizer.param_groups[0]["lr"]


def test_cosine_decayer():
    """Test the standalone CosineDecayer callable."""
    total_steps = 10
    decayer = CosineDecayer(total_steps=total_steps, n_cycles=2)
    val_beginning = decayer(0)
    val_middle = decayer(total_steps // 2)
    val_end = decayer(total_steps)
    assert 0 <= val_beginning <= 2, "CosineDecayer at step=0 is out of expected range."
    assert 0 <= val_middle <= 2, "CosineDecayer at mid-step is out of expected range."
    assert 0 <= val_end <= 2, "CosineDecayer at final step is out of expected range."


def test_linear_warmup(dummy_model_optimizer):
    """Test LinearWarmup scheduler for a small number of steps."""
    model, optimizer = dummy_model_optimizer
    total_steps = 10
    peak_step = 3
    scheduler = LinearWarmup(
        optimizer, total_steps, start_factor=0.01, peak_step=peak_step
    )
    initial_lr = current_lr(optimizer)
    lrs = []
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(current_lr(optimizer))
    assert lrs[0] < initial_lr, "LR did not start below initial LR during warmup."
    assert lrs[peak_step - 1] <= initial_lr, (
        "LR at peak_step should not exceed initial LR."
    )
    assert abs(lrs[-1] - initial_lr) < 1e-9, "LR after warmup should match initial LR."


def test_linear_warmup_cosine_annealing(dummy_model_optimizer):
    """Test LinearWarmupCosineAnnealing scheduler."""
    model, optimizer = dummy_model_optimizer
    total_steps = 10
    peak_step = 3
    scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        total_steps=total_steps,
        start_factor=0.01,
        end_lr=0.0,
        peak_step=peak_step,
    )
    initial_lr = current_lr(optimizer)
    lrs = []
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(current_lr(optimizer))
    assert lrs[0] < initial_lr, "LR not starting below initial during warmup."
    assert lrs[-1] <= 1e-5, (
        f"LR at final step is not near the end_lr=0.0, got {lrs[-1]}"
    )


def test_linear_warmup_cyclic_annealing(dummy_model_optimizer):
    """Test LinearWarmupCyclicAnnealing scheduler."""
    model, optimizer = dummy_model_optimizer
    total_steps = 10
    peak_step = 3
    scheduler = LinearWarmupCyclicAnnealing(
        optimizer,
        total_steps=total_steps,
        start_factor=0.01,
        peak_step=peak_step,
    )
    initial_lr = current_lr(optimizer)
    lrs = []
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(current_lr(optimizer))
    assert lrs[0] < initial_lr, (
        "Initial LR should be scaled by start_factor (< initial_lr)."
    )
    assert all(lr > 0 for lr in lrs), (
        "LR should stay positive throughout cyclic annealing."
    )
    assert lrs[-1] != lrs[peak_step - 1], (
        "LR did not change after warmup in cyclic decay."
    )


def test_linear_warmup_three_steps_annealing(dummy_model_optimizer):
    """Test LinearWarmupThreeStepsAnnealing scheduler."""
    model, optimizer = dummy_model_optimizer
    total_steps = 10
    peak_step = 3
    gamma = 0.5
    scheduler = LinearWarmupThreeStepsAnnealing(
        optimizer,
        total_steps=total_steps,
        start_factor=0.001,
        gamma=gamma,
        peak_step=peak_step,
    )
    initial_lr = current_lr(optimizer)
    lrs = []
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(current_lr(optimizer))
    assert lrs[0] < initial_lr, "Warmup did not start below initial LR."
    warmup_end_lr = lrs[peak_step - 1]
    final_lr = lrs[-1]
    assert final_lr < warmup_end_lr, (
        "ThreeStepsAnnealing did not reduce LR by final step."
    )
