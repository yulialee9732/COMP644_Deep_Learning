# 1.3 Feedforward Neural Networks — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.3-L3-Q01 — Network dimensions and parameters

$(xW_1+b_1)W_2+b_2=x(W_1W_2)+(b_1W_2+b_2)$. Repeating preserves affine form. Hidden nonlinearities allow input-dependent slopes and nonlinear decision boundaries.

## 1.3-L3-Q02 — Activation functions and gradients

Sigmoid has range (0,1), tanh (-1,1), and both saturate in their tails. ReLU has range [0,infinity), zero negative-side slope and unit positive slope. Leaky ReLU retains a small negative slope and is unbounded both ways. Rectifier kinks have no ordinary derivative at zero; implementations choose a convention.

## 1.3-L3-Q03 — Piecewise-linear representation

Under appropriate activation and compact-domain conditions, a sufficiently wide one-hidden-layer network can approximate a continuous target arbitrarily closely. It is an existence/capacity result; it does not provide an optimizer, practical width bound for every task, data guarantee, or generalization guarantee.
