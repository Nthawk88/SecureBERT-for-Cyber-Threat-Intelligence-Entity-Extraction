"""Ablation: contextual data augmentation for the minority ``Vulnerability`` class.

The proposal frames this as "LLM-based contextual data augmentation (template
CVE + auto-IOB labeling)". To keep the pipeline reproducible and runnable
offline, the default backend is a *template* generator that synthesises
realistic CVE-centric sentences and emits perfectly aligned IOB labels. An
optional LLM backend hook (:func:`generate_with_llm`) is provided for users who
want to plug in a real model later.

CRITICAL: augmented examples are added to the TRAIN split ONLY. Never augment
validation or test data.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from src.utils.labels import LABEL_LIST

# --- vocabulary pools for templating ----------------------------------------
_VENDORS = ["Microsoft", "Cisco", "Fortinet", "Apache", "VMware", "Atlassian",
            "Citrix", "Oracle", "Adobe", "GitLab", "Ivanti", "Palo Alto Networks"]
_PRODUCTS = ["Exchange Server", "IOS XE", "FortiOS", "Log4j", "vCenter",
             "Confluence", "NetScaler", "WebLogic", "Acrobat Reader",
             "Connect Secure", "PAN-OS", "Struts"]
_MALWARE = ["Cobalt Strike", "Emotet", "LockBit", "QakBot", "PlugX", "Mirai",
            "BlackCat", "Agent Tesla"]
_ORGS = ["CISA", "Mandiant", "Microsoft", "Unit 42", "Talos", "CrowdStrike"]

# (template string, per-token entity assignment via {placeholders})
# Each template is tokenised on whitespace; placeholders expand to one or more
# tokens whose labels are auto-assigned in IOB form.
_TEMPLATES: List[str] = [
    "{vendor} disclosed {cve} affecting {product} which is actively exploited in the wild .",
    "The {malware} operators leveraged {cve} to gain initial access to {product} .",
    "{org} warned that {cve} in {product} allows remote code execution .",
    "A critical vulnerability {cve} was patched by {vendor} in {product} this week .",
    "Threat actors chained {cve} with {malware} to compromise {product} deployments .",
    "Exploitation of {cve} targeting {product} has been observed by {org} .",
]


def _cve(rng: random.Random) -> str:
    year = rng.randint(2023, 2025)
    num = rng.randint(1000, 49999)
    return f"CVE-{year}-{num}"


def _emit(token_str: str, ent_type: Optional[str], tokens: List[str], tags: List[str]) -> None:
    """Append a (possibly multi-word) span with IOB tags."""
    parts = token_str.split()
    for i, part in enumerate(parts):
        tokens.append(part)
        if ent_type is None:
            tags.append("O")
        else:
            tags.append(f"{'B' if i == 0 else 'I'}-{ent_type}")


def generate_examples(
    num_examples: int,
    target_class: str = "Vulnerability",
    seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Generate ``num_examples`` synthetic (tokens, IOB-tags) pairs.

    Returns parallel lists suitable for appending to the train split.
    """
    rng = random.Random(seed)
    all_tokens: List[List[str]] = []
    all_tags: List[List[str]] = []
    for _ in range(num_examples):
        template = rng.choice(_TEMPLATES)
        fills = {
            "cve": (_cve(rng), "Vulnerability"),
            "vendor": (rng.choice(_VENDORS), "Organization"),
            "product": (rng.choice(_PRODUCTS), "System"),
            "malware": (rng.choice(_MALWARE), "Malware"),
            "org": (rng.choice(_ORGS), "Organization"),
        }
        tokens: List[str] = []
        tags: List[str] = []
        for raw in template.split():
            if raw.startswith("{") and raw.endswith("}"):
                key = raw[1:-1]
                value, ent = fills[key]
                _emit(value, ent, tokens, tags)
            else:
                _emit(raw, None, tokens, tags)
        all_tokens.append(tokens)
        all_tags.append(tags)
    return all_tokens, all_tags


def augment_train_dataset(
    dataset,
    num_augmented: int = 200,
    target_class: str = "Vulnerability",
    label_list: Optional[List[str]] = None,
    seed: int = 42,
    split_name: str = "train",
    llm_backend: Optional[Callable[[int, str, int], Tuple[List[List[str]], List[List[str]]]]] = None,
):
    """Append synthetic minority-class examples to a TRAIN :class:`datasets.Dataset`.

    Parameters
    ----------
    dataset:
        Train dataset with a ``ner_tags`` column (label ids).
    num_augmented:
        Number of synthetic sentences to generate.
    target_class:
        Minority class to enrich (default ``Vulnerability``).
    label_list:
        Label list for id mapping.
    seed:
        Determinism seed.
    split_name:
        Guard rail; must be ``"train"``.
    llm_backend:
        Optional callable ``(num, target_class, seed) -> (tokens, tags)`` to use a
        real LLM instead of the template generator.
    """
    if split_name != "train":
        raise ValueError(
            f"Augmentation must only touch the train split, got '{split_name}'."
        )

    labels = label_list or LABEL_LIST
    label2id = {lab: i for i, lab in enumerate(labels)}

    generator = llm_backend or generate_examples
    tokens, tags = generator(num_augmented, target_class, seed)
    tag_ids = [[label2id.get(t, label2id["O"]) for t in seq] for seq in tags]

    from datasets import Dataset, concatenate_datasets

    synth = Dataset.from_dict({"tokens": tokens, "ner_tags": tag_ids})
    # align feature schema to the source dataset
    synth = synth.cast(dataset.features)
    return concatenate_datasets([dataset, synth]).shuffle(seed=seed)


def generate_with_llm(num_examples: int, target_class: str, seed: int):  # pragma: no cover
    """Placeholder for a real LLM backend (not used by default).

    Implement a call to your preferred model here, returning ``(tokens, tags)``
    in the same IOB format as :func:`generate_examples`.
    """
    raise NotImplementedError(
        "LLM augmentation backend not configured. Use the default template "
        "generator or supply `llm_backend` to augment_train_dataset()."
    )
