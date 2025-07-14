import numpy as np
import torch


def _collapse_nested_dict(base, other):
    if type(base) in [list, tuple]:
        for i in range(len(base)):
            base[i] = _collapse_nested_dict(base[i], other[i])
        return base
    elif isinstance(base, dict):
        for key in base:
            base[key] = _collapse_nested_dict(base[key], other[key])
        return base
    else:
        base = torch.cat([base, other], 0)
        return base


class Collator:
    """Custom collate function that optionally builds an affinity (or “graph”) matrix based on a specified field."""

    def __init__(self, G_from=None):
        self.G_from = G_from

    def _flatten(self, x):
        batch = {}
        for name in x[0].keys():
            if type(x[0][name]) is list:
                flattened = sum([s[name] for s in x], [])
            else:
                flattened = [s[name] for s in x]
            if torch.is_tensor(flattened[0]):
                batch[name] = torch.stack(flattened, 0)
            elif type(flattened[0]) is dict:
                batch[name] = self._flatten(flattened)
            else:
                batch[name] = torch.from_numpy(np.array(flattened))
        return batch

    def __call__(self, samples):
        # samples = torch.utils.data.default_collate(samples)
        # contract the views if there are any
        single_view = torch.is_tensor(samples[0]["image"])
        if single_view:
            samples = torch.utils.data.default_collate(samples)
        else:
            samples = self._flatten(samples)

        if self.G_from is not None:
            t = samples[self.G_from]
            if t.ndim == 1 and t.dtype in [torch.long, torch.int]:
                G = (t[:, None].eq(t)).to(
                    device=samples["image"].device, dtype=samples["image"].dtype
                )
            else:
                G = t.flatten(1) @ t.flatten(1).T
            samples["G"] = G
        return samples

    @staticmethod
    def _test():
        indices = torch.randperm(50000)[:128]
        images = torch.randn((128, 3, 28, 28))
        labels = torch.randint(0, 10, size=(128,))
        # single view
        data = [
            dict(
                image=images[i],
                label=labels[i],
                idx=indices[i],
            )
            for i in range(128)
        ]
        collator = Collator(G_from="label")
        collected = collator(data)
        assert collected["image"].eq(images).all()
        assert collected["label"].eq(labels).all()
        assert collected["idx"].eq(indices).all()
        assert collected["G"].eq(labels[:, None] == labels).all()

        collator = Collator(G_from="idx")
        collected = collator(data)
        assert collected["G"].eq(indices[:, None] == indices).all()

        # multi-view
        data = [
            dict(
                image=[images[i] + torch.randn((3, 28, 28)) for _ in range(2)],
                label=[labels[i]] * 2,
                idx=[indices[i]] * 2,
            )
            for i in range(128)
        ]
        indices = torch.repeat_interleave(indices, 2)
        labels = torch.repeat_interleave(labels, 2)
        collator = Collator(G_from="label")
        collected = collator(data)
        assert collected["G"].eq(labels[:, None] == labels).all()

        collator = Collator(G_from="idx")
        collected = collator(data)
        assert collected["G"].eq(indices[:, None] == indices).all()
        return True
