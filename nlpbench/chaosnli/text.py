from __future__ import annotations

from .schema import ChaosNLIRawItem



def format_chaosnli_text(item: ChaosNLIRawItem) -> str:
    """Return the text fed to the embedding encoder."""
    return f"Premise: {item.premise}\nHypothesis: {item.hypothesis}"
