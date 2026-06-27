"""Compute-device detection shared across cover + training.

Resolves DEVICE=auto into the best available backend so the same code runs on
a local Apple-Silicon Mac (mps), a CUDA GPU server (cuda), or CPU fallback.
"""
import os


def resolve_device(preference: str | None = None) -> str:
    """Return one of: 'cuda', 'mps', 'cpu'.

    preference: 'auto' | 'cuda' | 'mps' | 'cpu' | None. Falls back to env DEVICE,
    then 'auto'. A forced device that isn't available degrades to the best one
    that is, with no exception (so the app still runs, just slower).
    """
    pref = (preference or os.getenv("DEVICE") or "auto").lower()

    import torch

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if pref in ("cuda", "mps", "cpu"):
        if pref == "cuda" and has_cuda:
            return "cuda"
        if pref == "mps" and has_mps:
            return "mps"
        if pref == "cpu":
            return "cpu"
        # forced device unavailable -> fall through to auto

    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def device_report() -> dict:
    """Human-facing summary for the /api/gpu endpoint and startup logs."""
    import torch

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    resolved = resolve_device()
    return {
        "resolved": resolved,
        "cuda": has_cuda,
        "mps": has_mps,
        "torch": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if has_cuda else None,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(device_report(), indent=2))
