from __future__ import annotations

"""Constants used by the ChaosNLI loader."""

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

SHORT_LABEL_TO_LONG = {
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
}

SUBSET_TO_FILENAME = {
    "snli": "chaosNLI_snli.jsonl",
    "mnli": "chaosNLI_mnli_m.jsonl",
}

DEFAULT_CHAOSNLI_URL = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"
