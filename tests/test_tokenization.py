"""Unit tests for IOB subword label alignment (no network / HF download needed)."""

from typing import List

from src.preprocessing.tokenization import tokenize_and_align_labels


class _FakeEncoding(dict):
    """Mimics HuggingFace BatchEncoding's ``word_ids`` API."""

    def __init__(self, word_ids_per_example: List[List]):
        super().__init__()
        self._word_ids = word_ids_per_example

    def word_ids(self, batch_index: int = 0):
        return self._word_ids[batch_index]


class _FakeTokenizer:
    """Splits each word into a deterministic number of subwords.

    ``subword_counts[i]`` controls how many subwords word ``i`` produces. Adds a
    leading and trailing special token (word_id = None) like real tokenizers.
    """

    def __init__(self, subword_counts: List[int]):
        self.subword_counts = subword_counts

    def __call__(self, tokens, **kwargs):
        word_ids = [None]  # [CLS]
        for word_idx, _ in enumerate(tokens[0]):
            count = self.subword_counts[word_idx]
            word_ids.extend([word_idx] * count)
        word_ids.append(None)  # [SEP]
        enc = _FakeEncoding([word_ids])
        enc["input_ids"] = [list(range(len(word_ids)))]
        return enc


def test_first_subword_gets_label_rest_ignored():
    # 3 words; word 1 splits into 3 subwords, others into 1.
    tokenizer = _FakeTokenizer(subword_counts=[1, 3, 1])
    examples = {
        "tokens": [["CVE-2024-1234", "exploited", "Apache"]],
        "ner_tags": [[9, 0, 7]],  # B-Vulnerability, O, B-Organization
    }
    out = tokenize_and_align_labels(examples, tokenizer, max_length=128)
    labels = out["labels"][0]

    # [CLS]=-100, w0=9, w1 first=0 then two -100, w2=7, [SEP]=-100
    assert labels == [-100, 9, 0, -100, -100, 7, -100]


def test_label_all_tokens_propagates_label():
    tokenizer = _FakeTokenizer(subword_counts=[2, 1])
    examples = {
        "tokens": [["LockBit", "ransomware"]],
        "ner_tags": [[3, 0]],  # B-Malware, O
    }
    out = tokenize_and_align_labels(
        examples, tokenizer, max_length=128, label_all_tokens=True
    )
    labels = out["labels"][0]
    # [CLS]=-100, w0 first=3, w0 second=3 (propagated), w1=0, [SEP]=-100
    assert labels == [-100, 3, 3, 0, -100]


def test_special_and_padding_tokens_ignored():
    tokenizer = _FakeTokenizer(subword_counts=[1, 1])
    examples = {"tokens": [["a", "b"]], "ner_tags": [[0, 0]]}
    out = tokenize_and_align_labels(examples, tokenizer, max_length=128)
    labels = out["labels"][0]
    assert labels[0] == -100 and labels[-1] == -100
