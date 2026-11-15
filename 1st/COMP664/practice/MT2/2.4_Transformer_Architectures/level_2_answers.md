# 2.4 Transformer Architectures — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.4-L2-Q01 — Transformer blocks and parameter counts

Blocks contribute $12(4)(32^2)=49,152$; tied vocabulary contributes 6400; total 55,552. Untying adds another $Vd=6400$, giving 61,952.

## 2.4-L2-Q02 — Positions, embeddings, and logits

Input E is $500\times24$; output multiplication uses $E^\top:24\times500$. The shared table has 12,000 parameters. Three subwords occupy three positions and require separate embeddings and positional treatment, rather than one position.

## 2.4-L2-Q03 — Causal masks and cross-attention

Causal rows are (1,0,0,0), (1/2,1/2,0,0), (1/3,1/3,1/3,0), (1/4,1/4,1/4,1/4). Cross-attention scores are $3\times5$, with target-query rows and source-key columns.

## 2.4-L2-Q04 — Architecture and objective selection

BERT predicts selected corrupted tokens using bidirectional context; original BERT also used next-sentence prediction. T5 expresses tasks as text input/output and can use task prefixes. GPT uses causal next-token prediction. Masked-token loss $-\ln0.5-\ln0.25=\ln8\approx2.0794$.
