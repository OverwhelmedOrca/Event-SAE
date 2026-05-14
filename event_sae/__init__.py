"""event_sae — lazy top-level package.

Subpackages have different dependency sets (e.g. `event_sae.sae` needs
`dictionary_learning`, `event_sae.openpi` needs the openpi fork). To
avoid forcing every consumer to install everything, the top-level
exports are loaded lazily on attribute access. Usage:

  from event_sae import SAE                  # loads event_sae.sae lazily
  from event_sae.openpi.eval import runner   # never touches event_sae.sae
"""

__all__ = ["SAE", "BatchTopKSAE", "SAETrainConfig", "train_sae"]


def __getattr__(name: str):
    if name in ("SAE", "BatchTopKSAE"):
        from event_sae import sae as _sae

        return getattr(_sae, name)
    if name in ("SAETrainConfig", "train_sae"):
        from event_sae import train as _train

        return getattr(_train, name)
    raise AttributeError(f"module 'event_sae' has no attribute {name!r}")
