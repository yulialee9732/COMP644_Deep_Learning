# 2.4 Transformer Architectures — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.4-L3-Q01 — Transformer blocks and parameter counts

Attention mixes token information with Q/K/V/O projections totaling $4d^2$. The FFN nonlinearly transforms each token independently with $d\times4d$ and $4d\times d$ matrices totaling $8d^2$. A block reuses its maps at every position; ordinary stacked blocks have distinct parameters. Norms, biases, and positions are additional.

## 2.4-L3-Q02 — Positions, embeddings, and logits

Permuting input rows permutes Q/K/V; scores permute along rows and columns, row-softmax follows that permutation, and outputs permute accordingly. It does not identify absolute sequence order. Positional representations and position-dependent masks such as causality introduce order sensitivity. Position-wise FFNs alone do not supply order.

## 2.4-L3-Q03 — Causal masks and cross-attention

Each training position represents a known prefix token and predicts the next token; the mask blocks later positions, including the position containing its target. Computation across positions can run in parallel without exposing future targets. Self-attention Q/K/V all come from decoder states; cross-attention Q comes from decoder states, K/V from encoder outputs.

## 2.4-L3-Q04 — Architecture and objective selection

Encoders use bidirectional supplied-input context and can train masked prediction, then attach task heads. Encoder–decoders combine a bidirectional source encoder, causal target decoder, and cross-attention; a class label can be a generated text string. Decoder-only models train causal next-token prediction and condition on a prompt. All can support multiple tasks with appropriate training; architecture alone does not determine competence.
