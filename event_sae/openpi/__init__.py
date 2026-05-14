"""event_sae openpi half — π₀.₅ activation collection, eval, and intervention.

The `activations` submodule requires the openpi-event-sae fork (the JAX
side); the `eval` submodule only needs `openpi-client` + LIBERO. Both
are loaded lazily so the LIBERO sim venv (which has neither openpi nor
dictionary_learning) can still `from event_sae.openpi.eval import ...`.
"""

__all__ = ["TopKActivationCollector"]


def __getattr__(name: str):
    if name == "TopKActivationCollector":
        from event_sae.openpi import activations as _activations

        return _activations.TopKActivationCollector
    raise AttributeError(f"module 'event_sae.openpi' has no attribute {name!r}")
