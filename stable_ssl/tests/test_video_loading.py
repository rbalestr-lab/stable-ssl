def test_clip_extract():
    import optimalssl

    dataset = optimalssl.data.HFDataset(
        path="shivalikasingh/video-demo",
        split="train",
        trust_remote_code=True,
        transform=optimalssl.data.transforms.RandomContiguousTemporalSampler(
            source="video", target="frames", num_frames=10
        ),
    )
    assert "video" in dataset[0]
    assert "frames" in dataset[0]
    # print(dataset["video"])
    assert dataset[0]["frames"].ndim == 4
    assert dataset[0]["frames"].size(0) == 10
    assert dataset[0]["frames"].size(1) == 3


def test_clip_dataset():
    import optimalssl
    import torch

    dataset = optimalssl.data.HFDataset(
        path="shivalikasingh/video-demo",
        split="train",
        trust_remote_code=True,
        transform=optimalssl.data.transforms.Compose(
            optimalssl.data.transforms.RandomContiguousTemporalSampler(
                source="video", target="video", num_frames=10
            ),
            optimalssl.data.transforms.Resize(
                (128, 128), source="video", target="video"
            ),
        ),
    )
    dataset = torch.utils.data.ConcatDataset([dataset for _ in range(10)])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for data in loader:
        assert "video" in data
        assert "frames" not in data
        print(data["idx"])
        assert data["video"].shape == (4, 10, 3, 128, 128)


def test_embedding_from_image():
    import optimalssl
    import torch
    import torchvision

    dataset = optimalssl.data.HFDataset(
        path="shivalikasingh/video-demo",
        split="train",
        trust_remote_code=True,
        transform=optimalssl.data.transforms.Compose(
            optimalssl.data.transforms.RandomContiguousTemporalSampler(
                source="video", target="video", num_frames=10
            ),
            optimalssl.data.transforms.Resize(
                (128, 128), source="video", target="video"
            ),
            optimalssl.data.transforms.ToImage(
                scale=False,
                mean=[0, 0, 0],
                std=[255, 255, 255],
                source="video",
                target="video",
            ),
        ),
    )
    dataset = torch.utils.data.ConcatDataset([dataset for _ in range(10)])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    embedding = optimalssl.utils.ImageToVideoEncoder(torchvision.models.resnet18())
    for data in loader:
        features = embedding(data["video"])
        assert features.shape == (4, 10, 1000)


if __name__ == "__main__":
    test_clip_extract()
    test_clip_dataset()
    test_embedding_from_image()
