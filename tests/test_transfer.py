import torch

from plantdisease.models import build_transfer_model, count_parameters, count_total_parameters


def test_transfer_output_shape():
    model = build_transfer_model(num_classes=39, pretrained=False).eval()
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 39)


def test_frozen_backbone_trains_only_the_head():
    model = build_transfer_model(39, pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]
    assert count_parameters(model) < count_total_parameters(model)
