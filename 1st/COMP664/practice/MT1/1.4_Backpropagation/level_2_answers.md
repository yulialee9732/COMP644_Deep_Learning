# 1.4 Backpropagation — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.4-L2-Q01 — Multiple paths in a computational graph

$u=9$, $v=6$, $L=54$. Contributions are $v(2x)=36$ and $u(2)=18$; total 54. Directly $L=2x^3$ gives the same derivative.

## 1.4-L2-Q02 — Forward and backward passes

$z_1=h=6$, $s=6$, loss 8. $\delta_2=4$, $g_{w_2}=24$, $g_{w_1}=12$. Simultaneous updates from old weights give $w_1=1.88$, $w_2=0.76$.

## 1.4-L2-Q03 — Output derivatives and deep gradients

Using the Jacobian and chain rule gives $(0.6,0.1,-0.7)$. The derivative product is $0.25^6=1/4096\approx0.00024414$, illustrating contraction when other factors are one.
