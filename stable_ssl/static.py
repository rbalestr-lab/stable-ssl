_EMBEDDING_DIMENSION = {
    "resnet50.a1_in1k": 2048,
    "hf_hub:edadaltocg/resnet18_cifar10": 512,
    "hf_hub:edadaltocg/resnet50_cifar10": 2048,
}


def embedding_dim(model_name: str) -> int:
    # resnets
    if "resnet18" in model_name:
        return 512
    elif "resnet34" in model_name:
        return 1024
    elif "resnet" in model_name:
        return 2048
    # ViTs
    if "_t" in model_name or "_b" in model_name:
        return 1024
    else:
        return 2048
    return _EMBEDDING_DIMENSION[model_name]
