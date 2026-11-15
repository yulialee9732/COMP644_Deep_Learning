# 1.8 Representation Learning & Embeddings — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.8-L3-Q01 — Embedding lookup and context supervision

One-hot vectors identify discrete items but encode no learned similarity; embedding lookup selects a dense trainable row. Subwords reuse pieces and reduce whole-word vocabulary pressure, while potentially increasing sequence length. Smaller windows often emphasize local syntactic relations, larger windows topical association; these are tendencies, not guarantees.

## 1.8-L3-Q02 — Full-softmax skip-gram

$\ell=-v_o^\top u_c+\ln\sum_we^{v_w^\top u_c}$. Differentiate to get $-v_o+\sum_wp(w|c)v_w$. Descent moves toward the observed context relative to the model expectation. Normalization sums over the vocabulary, costing about $O(Vd)$ per pair for direct evaluation.

## 1.8-L3-Q03 — Negative sampling

Negative sampling trains binary discrimination of observed versus sampled noise pairs. It uses the positive and K negative scores, about $O(Kd)$, rather than all vocabulary scores. It changes the objective and does not directly produce the same normalized conditional distribution as full softmax.

## 1.8-L3-Q04 — Similarity, transfer, and context

Analogy offsets are empirical patterns, not universal semantic laws; test multiple relations and nearest-neighbor choices. A 2D projection distorts high-dimensional geometry, so visual clusters alone do not prove semantics. Static lookups give each token one row; contextual representations depend on the surrounding sequence.
