"""CTI Platform — Improving Cyber Threat Intelligence using SecureBERT.

Two complementary contributions:
  1. Higher entity-extraction precision (esp. the ``Vulnerability`` class) via a
     domain-adapted language model + class-imbalance mitigation (Focal Loss).
  2. A lightweight, heuristic IoC co-occurrence correlation layer that turns
     isolated indicators into a relational "campaign picture".
"""

__version__ = "0.1.0"
