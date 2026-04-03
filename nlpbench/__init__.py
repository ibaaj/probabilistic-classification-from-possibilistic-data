"""Public package surface for the NLP benchmark helpers.

This package currently exposes:

- generic embedding-cache utilities in ``nlpbench.embeddings``
- deterministic sampling helpers in ``nlpbench.sampling``
- the ChaosNLI loader in ``nlpbench.chaosnli``

For convenience, the two main ChaosNLI entry points are re-exported at the
package root.
"""

from . import chaosnli, embeddings, sampling
from .chaosnli import add_chaosnli_data_args, load_chaosnli_splits

__all__ = [
    "chaosnli",
    "embeddings",
    "sampling",
    "add_chaosnli_data_args",
    "load_chaosnli_splits",
]