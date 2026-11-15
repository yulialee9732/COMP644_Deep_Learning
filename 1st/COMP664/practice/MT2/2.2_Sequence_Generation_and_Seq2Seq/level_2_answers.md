# 2.2 Sequence Generation, Teacher Forcing, Sampling, and Seq2Seq Models — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.2-L2-Q01 — Autoregressive generation and teacher forcing

Inputs BOS,a,blue,bird; targets a,blue,bird,EOS. Joint probability 0.05; summed loss $\ln20\approx2.9957$; mean loss $\ln20/4\approx0.7489$.

## 2.2-L2-Q02 — Greedy decoding and sampling

Top-2 gives $(0.625,0.375,0,0)$. Top-p retains the first three (cumulative 0.95), giving $(10/19,6/19,3/19,0)$. At T=2 exponentials are (2,1), giving $(2/3,1/3)$.

## 2.2-L2-Q03 — Encoder–decoder attention

Weights $(1/2,1/4,1/4)$; context $(1.5,1)$. A final-state-only encoder compresses the source into one fixed-size state; attention gives the decoder direct access to all encoder states with step-specific weighting.
