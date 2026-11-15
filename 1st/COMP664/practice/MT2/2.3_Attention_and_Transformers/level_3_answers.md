# 2.3 Attention & Transformers — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.3-L3-Q01 — Scaled dot-product attention

For $X:L\times d$, $W_Q,W_K:d\times d_k$, $W_V:d\times d_v$, Q and K are $L\times d_k$, V is $L\times d_v$, scores are $L\times L$, and output is $L\times d_v$. Queries seek matches, keys supply matches, values carry retrieved content. Under roughly independent unit-variance coordinates, dot-product variance grows with $d_k$; dividing by its square root controls scale and softmax saturation.

## 2.3-L3-Q02 — Multi-head dimensions and projections

Heads have distinct learned Q/K/V maps and can learn different mixing patterns. At fixed d with head width d/H, total projected width remains d, giving about $4d^2$ weights. Roles are learned and need not be distinct, stable, or interpretable; a plausible pattern is not proof of exclusive function.

## 2.3-L3-Q03 — Information paths and attention complexity

RNN states form a temporal chain; self-attention can mix distant positions directly and compute many known-input positions in parallel. During ordinary autoregressive generation each selected token changes the next prefix, so new-token generation remains sequential even when the operations inside each pass are parallel.
