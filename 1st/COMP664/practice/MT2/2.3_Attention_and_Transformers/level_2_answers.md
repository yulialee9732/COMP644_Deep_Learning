# 2.3 Attention & Transformers — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.3-L2-Q01 — Scaled dot-product attention

Dot products are 0 and $\sqrt2\ln3$. Dividing by $\sqrt2$ gives $(0,\ln3)$; weights $(1/4,3/4)$; output $(0.25,3)$.

## 2.3-L2-Q02 — Multi-head dimensions and projections

Head width 32; each per-head projection $192\times32$; concatenated width 192; $W_O:192\times192$; total $4(192^2)=147,456$. With 12 heads each width is 16, and total projection-weight count stays the same under this convention.

## 2.3-L2-Q03 — Information paths and attention complexity

At L=100: $8(100^2)=80,000$ entries. At L=400: $8(400^2)=1,280,000$. Length increases fourfold; score storage increases sixteenfold, with other dimensions fixed.
