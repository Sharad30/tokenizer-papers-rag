# Phase 2 probe: dilution and truncation

Paper: `acl:P16-1162` · anchor: "initialize the symbol vocabulary with the character vocabulary" · model limit: 512 tokens

Question: How does BPE build its subword vocabulary by merging characters?

Answer paragraph alone: 156 tokens, similarity 0.800

| Tokens | Answer first | Answer last |
|---|---|---|
| 128 | 0.779 | 0.758 |
| 192 | 0.801 | 0.788 |
| 255 | 0.800 | 0.770 |
| 377 | 0.802 | 0.746 |
| 448 | 0.800 | 0.758 |
| 510 | 0.796 | 0.702 |
| 576 | 0.796 | 0.708 |
| 640 | 0.796 | 0.715 |
| 768 | 0.796 | 0.680 |
| 1024 | 0.796 | 0.685 |
