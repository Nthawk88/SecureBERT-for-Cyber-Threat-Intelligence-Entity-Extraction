"""Tests for CyNER CoNLL parsing and class-distribution counting."""

from src.preprocessing.cyner_loader import class_distribution, read_conll
from src.utils.labels import LABEL_LIST

SAMPLE_CONLL = """\
-DOCSTART- O

The O
APT28 B-Organization
group O
exploited O
CVE-2024-1234 B-Vulnerability
in O
Microsoft B-Organization
Exchange B-System
Server I-System

Emotet B-Malware
dropped O
"""


def test_read_conll_parses_sentences(tmp_path):
    path = tmp_path / "sample.conll"
    path.write_text(SAMPLE_CONLL, encoding="utf-8")
    tokens, tags = read_conll(path, valid_labels=LABEL_LIST)

    assert len(tokens) == 2
    assert tokens[0][1] == "APT28"
    assert tags[0][1] == "B-Organization"
    # multi-token System entity
    assert tags[0][-2:] == ["B-System", "I-System"]
    assert tokens[1] == ["Emotet", "dropped"]
    assert tags[1] == ["B-Malware", "O"]


def test_read_conll_coerces_unknown_labels(tmp_path):
    path = tmp_path / "bad.conll"
    path.write_text("foo B-NotARealClass\nbar O\n", encoding="utf-8")
    _, tags = read_conll(path, valid_labels=LABEL_LIST)
    assert tags[0] == ["O", "O"]


def test_class_distribution_entity_level(tmp_path):
    path = tmp_path / "sample.conll"
    path.write_text(SAMPLE_CONLL, encoding="utf-8")
    _, tags = read_conll(path, valid_labels=LABEL_LIST)
    dist = class_distribution(tags, entity_level=True)
    # entities: 2x Organization, 1x Vulnerability, 1x System, 1x Malware
    assert dist["Organization"] == 2
    assert dist["Vulnerability"] == 1
    assert dist["System"] == 1
    assert dist["Malware"] == 1
