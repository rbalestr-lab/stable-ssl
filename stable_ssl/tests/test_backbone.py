import pytest
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
)


@pytest.mark.parametrize(
    "name",
    [
        "alexnet",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "inception_v3",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_1_6gf",
        "regnet_y_3_2gf",
        "regnet_y_8gf",
        "regnet_y_16gf",
        "regnet_y_32gf",
        "regnet_x_400mf",
        "regnet_x_800mf",
        "regnet_x_1_6gf",
        "regnet_x_3_2gf",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ],
)
def test_torchvision_embedding_dim(name):
    import torchvision

    import stable_ssl as ossl

    if "vit" in name:
        shape = (10, 3, 224, 224)
    else:
        shape = (10, 3, 512, 512)

    module = torchvision.models.__dict__[name]()
    ossl.backbone.set_embedding_dim(
        module,
        dim=20,
        expected_input_shape=shape,
        expected_output_shape=(shape[0], 20),
    )


@pytest.mark.parametrize(
    "name,method,shape",
    [
        ("microsoft/resnet-18", AutoModelForImageClassification, 224),
        ("timm/swin_tiny_patch4_window7_224.ms_in1k", AutoModel, 224),
    ],
)
def test_hf_embedding_dim(name, method, shape):
    import torch

    import stable_ssl as ossl

    module = method.from_pretrained(name)

    module = ossl.backbone.set_embedding_dim(
        module,
        dim=20,
        expected_input_shape=(10, 3, shape, shape),
        expected_output_shape=(10, 20),
    )
    x = torch.empty((10, 3, shape, shape), device="meta")
    out = module.to("meta")(x)
    if isinstance(out, tuple):
        assert out[0].shape == (10, 20)
    elif hasattr(out, "logits"):
        assert out["logits"].shape == (10, 20)
    else:
        assert out.shape == (10, 20)
