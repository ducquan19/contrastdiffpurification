"""Clean / robust accuracy of a model on an in-memory tensor dataset."""
import torch


def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda')):
    """Top-1 accuracy of ``model`` on ``(x_orig, y_orig)``, evaluated in batches of ``bs``.

    Note: only ``len(x_orig) // bs`` full batches are evaluated while the count is divided by
    ``len(x_orig)``, so a trailing partial batch counts as misclassified.
    """
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()

    if isinstance(acc, torch.Tensor):
        acc = acc.item()
    return acc / x_orig.shape[0]


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval of a proportion ``k / n`` (95% for z = 1.96); ``(nan, nan)`` if n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return centre - half, centre + half
