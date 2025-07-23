import torch


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, x):
    return a + x * (b - a)


def _grad(hash, x, y):
    h = hash & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _perlin(x, y, permutation):
    xi = x.to(torch.int32) & 255
    yi = y.to(torch.int32) & 255
    xf = x - x.to(torch.int32)
    yf = y - y.to(torch.int32)
    u = _fade(xf)
    v = _fade(yf)
    aa = permutation[permutation[xi] + yi]
    ab = permutation[permutation[xi] + yi + 1]
    ba = permutation[permutation[xi + 1] + yi]
    bb = permutation[permutation[xi + 1] + yi + 1]
    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)


def generate_perlin_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2.0):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 256
    )
    permutation = torch.arange(256, dtype=torch.int32)
    permutation = permutation[torch.randperm(256)]
    permutation = torch.cat([permutation, permutation])
    noise = torch.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        for i in range(d[0]):
            for j in range(d[1]):
                noise[i :: d[0], j :: d[1]] += amplitude * _perlin(
                    grid[i :: d[0], j :: d[1], 0] * frequency,
                    grid[i :: d[0], j :: d[1], 1] * frequency,
                    permutation,
                )
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    noise /= max_amplitude
    return noise
