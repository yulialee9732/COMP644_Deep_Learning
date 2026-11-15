# 1.5 Learning Rates and Regularization — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.5-L2-Q01 — Learning-rate stability

Multipliers are 0.5, -0.5, -1, -1.5; first weights are 1, -1, -2, -3. The first converges without oscillation, the second converges with oscillation, the third cycles at fixed magnitude, and the fourth diverges with oscillation.

## 1.5-L2-Q02 — Momentum, RMSProp, and Adam

$m_1=-0.6$, $v_1=0.9$, corrected moments -3 and 9. Adam change is +0.02; SGD change is +0.06. Adam rescales by the second-moment estimate.

## 1.5-L2-Q03 — Regularization and dropout

L2 gradient $1+0.2(-3)=0.4$ gives -3.2. L1 gradient $1-0.2=0.8$ gives -3.4. Masked output is $(3/0.75,0)=(4,0)$. The dropped path has zero backward gradient.
