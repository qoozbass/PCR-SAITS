import numpy as np
import torch

from pcrsaits.model import PCRExampleDataset, PCRResidualNet


def test_example_dataset_forces_float32():
    ds = PCRExampleDataset(
        np.ones((2, 10), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.array([True, False]),
    )
    assert ds.features.dtype == torch.float32
    assert ds.targets.dtype == torch.float32
    assert ds.base_errors.dtype == torch.float32
    assert ds.easy_mask.dtype == torch.float32


def test_v71_parameterization_and_zero_delta_head():
    torch.manual_seed(1)
    model = PCRResidualNet(input_dim=10, hidden_dim=64)

    assert model.backbone[0].in_features == 10
    assert model.backbone[0].out_features == 64
    assert model.backbone[2].in_features == 64
    assert model.backbone[2].out_features == 64

    assert torch.count_nonzero(model.delta_head.weight) == 0
    assert torch.count_nonzero(model.delta_head.bias) == 0
    assert torch.count_nonzero(model.mask_head.weight) == 0
    torch.testing.assert_close(
        model.mask_head.bias,
        torch.full_like(model.mask_head.bias, -1.0),
    )


def test_zero_initialization_makes_proposed_delta_zero_and_mask_one():
    model = PCRResidualNet(10)
    x = torch.randn(5, 10)
    delta, mask = model(x, direct_residual=True)
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(mask, torch.ones_like(mask))


def test_masked_residual_initial_mask_is_sigmoid_minus_one():
    model = PCRResidualNet(10)
    x = torch.randn(3, 10)
    _, mask = model(x, direct_residual=False)
    expected = torch.sigmoid(torch.tensor(-1.0))
    torch.testing.assert_close(
        mask,
        torch.full_like(mask, expected),
    )


def test_residual_is_bounded_by_four():
    model = PCRResidualNet(10)
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.fill_(100.0)
    delta, _ = model(torch.zeros(4, 10), direct_residual=True)
    assert torch.all(delta <= 4.0)
    assert torch.all(delta > 3.999)


def test_parameter_count_matches_legacy_and_active_proposed_count():
    model = PCRResidualNet(10, 64)
    total = sum(p.numel() for p in model.parameters())
    mask_head = sum(p.numel() for p in model.mask_head.parameters())
    assert total == 4994
    assert total - mask_head == 4929
