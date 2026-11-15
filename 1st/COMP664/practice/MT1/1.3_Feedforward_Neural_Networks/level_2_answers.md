# 1.3 Feedforward Neural Networks — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.3-L2-Q01 — Network dimensions and parameters

$W_1:5\times6$, $b_1:6$, $H:10\times6$, $W_2:6\times3$, $b_2:3$, output $10\times3$. Count $30+6+18+3=57$. Use softmax for three mutually exclusive classes.

## 1.3-L2-Q02 — Activation functions and gradients

Both rectifiers output 2 and propagate -4. Sigmoid derivatives are $0.5(0.5)=0.25$ and $0.99(0.01)=0.0099$; saturation greatly reduces the local gradient.

## 1.3-L2-Q03 — Piecewise-linear representation

$g=0$ for $x\le1$; $g=2x-2$ for $1<x<3$; $g=x+1$ for $x\ge3$. Values are 0, 2, 5. Slopes are 0, 2, 1.
