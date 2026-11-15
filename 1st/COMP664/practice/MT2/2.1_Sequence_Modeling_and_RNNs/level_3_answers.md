# 2.1 Sequence Modeling and RNNs — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.1-L3-Q01 — Markov histories and sequence probability

Markov models explicitly condition on a fixed number of recent tokens, leading to combinatorially many discrete contexts. RNNs compress a prefix into a shared continuous state. They can in principle use more distant information, but finite state size and optimization mean useful long-term retention is not guaranteed.

## 2.1-L3-Q02 — Recurrent forward computation

Token labeling applies a shared head to each $h_t$; sequence classification can use a head on final $h_T$ (or a suitable pooling scheme). Each $h_t$ depends on $h_{t-1}$, so ordinary recurrent state computation must advance in time order. Parameters are shared across sequence length.

## 2.1-L3-Q03 — Backpropagation through time

Unrolling creates one computational node per state update but reuses the same parameters. Reverse-mode differentiation follows the chain back in time and sums each use of a parameter. Contracting Jacobian products weaken distant credit assignment; expanding products can destabilize it. Sequential computation is a separate limitation from these gradient effects.
