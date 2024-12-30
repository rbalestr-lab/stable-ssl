import pytest
import torch
import torch.nn as nn

from stable_ssl.modules import (
    MLP,
    ConvMixer,
    Resnet9,
    TeacherStudentModule,
    load_backbone,
)


@pytest.fixture
def setup_models():
    student_model = Resnet9(num_classes=10)
    teacher_model = TeacherStudentModule(student_model, ema_coefficient=0.99)
    return student_model, teacher_model


def test_load_backbone_resnet9():
    model = load_backbone("resnet9", num_classes=10)
    assert isinstance(model, Resnet9), "The model should be an instance of Resnet9"
    assert isinstance(model.fc, nn.Linear), "The last layer should be a Linear layer"


def test_load_backbone_convmixer():
    model = load_backbone("ConvMixer", num_classes=10)
    assert isinstance(model, ConvMixer), "The model should be an instance of ConvMixer"
    assert isinstance(model.fc, nn.Linear), "The last layer should be a Linear layer"


def test_load_backbone_torchvision_model():
    model = load_backbone("resnet18", num_classes=10)
    assert isinstance(model, nn.Module), "The model should be a PyTorch module"
    assert isinstance(model.fc, nn.Linear), "The last layer should be a Linear layer"


def test_load_backbone_low_resolution():
    model = load_backbone("resnet9", num_classes=10, low_resolution=True)
    assert isinstance(
        model.conv1, nn.Conv2d
    ), "The first conv layer should be a Conv2d layer"
    assert isinstance(model.maxpool, nn.Identity), "Maxpool should be an Identity layer"


def test_teacher_student_module_initialization(setup_models):
    student_model, teacher_model = setup_models
    assert isinstance(
        teacher_model.teacher, nn.Module
    ), "The teacher should be a model instance"
    for param in teacher_model.teacher.parameters():
        assert (
            param.requires_grad is False
        ), "The teacher model should not require gradients"


def test_mlp_forward():
    model = MLP([128, 64, 32, 10])
    x = torch.randn(8, 128)
    output = model(x)
    assert output.shape == (
        8,
        10,
    ), f"Output shape should be (8, 10), but got {output.shape}"


def test_resnet9_forward():
    model = Resnet9(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    output = model(x)
    assert output.shape == (
        8,
        10,
    ), f"Output shape should be (8, 10), but got {output.shape}"


def test_convmixer_forward():
    model = ConvMixer(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    output = model(x)
    assert output.shape == (
        8,
        10,
    ), f"Output shape should be (8, 10), but got {output.shape}"
