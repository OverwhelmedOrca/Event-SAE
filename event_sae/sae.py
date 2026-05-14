"""SAE classes used by Event-SAE — thin re-exports from `dictionary_learning`."""

from dictionary_learning import AutoEncoder as SAE
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

__all__ = ["SAE", "BatchTopKSAE"]
