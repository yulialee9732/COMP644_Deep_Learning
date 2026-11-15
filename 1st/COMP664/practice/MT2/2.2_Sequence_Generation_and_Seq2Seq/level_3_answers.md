# 2.2 Sequence Generation, Teacher Forcing, Sampling, and Seq2Seq Models — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.2-L3-Q01 — Autoregressive generation and teacher forcing

Training conditions each prediction on the ground-truth prefix; generation conditions on its selected prefix. A mistaken token changes subsequent context, so later predictions may receive unusual or inconsistent prefixes. Errors can accumulate, although this does not imply every error causes failure.

## 2.2-L3-Q02 — Greedy decoding and sampling

Greedy chooses an argmax deterministically subject to ties. Sampling draws according to probabilities. Top-k truncates to a fixed count; nucleus truncates by cumulative mass; both renormalize and alter the distribution. Positive temperature rescales logits: lower values sharpen, higher flatten. Diversity changes do not by themselves guarantee quality.

## 2.2-L3-Q03 — Encoder–decoder attention

Each decoder step scores all source states, so successive target positions can emphasize source positions in a different order. Rows index target steps, columns source positions; each row sums to one. Translation loss differentiates through attention, allowing soft alignments to emerge without word-alignment supervision.
