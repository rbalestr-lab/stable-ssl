from stable_ssl.data import (
    HuggingFaceDataset,
    MultiViewSampler,
)


def test_multiview_sampler():
    transforms = [lambda x: x * 2, lambda x: x + 1]
    sampler = MultiViewSampler(transforms)

    output = sampler(5)
    assert output == [10, 6]

    sampler = MultiViewSampler([lambda x: x + 3])
    output = sampler(5)
    assert output == 8


def test_huggingface_dataset():
    hf_dataset = HuggingFaceDataset(
        path="mnist",
        split="train[:1000]",
    )

    assert len(hf_dataset) == 1000
    sample = hf_dataset[0]
    assert isinstance(sample, dict)
    assert len(sample) == 2
