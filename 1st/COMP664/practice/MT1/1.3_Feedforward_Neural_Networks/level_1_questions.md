# 1.3 Feedforward Neural Networks — Level 1

Sources: [section summary](../../../MT1/1.3%20FNN/1.3_Feedforward_Neural_Networks.md) and [study notes](../../../MT1/1.3%20FNN/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.3-L1-Q01 — Network dimensions and parameters

A network has dimensions 3→4→2 and a batch of five rows. Give all weight, bias, hidden, and output shapes and count trainable parameters.

**Formula support:** $H=\phi(XW_1+b_1)$, $Y=g(HW_2+b_2)$; parameters $=dh+h+hC+C$.

**Step-by-step solution**

1. $W_1:3\times4$, $b_1:4$, $W_2:4\times2$, $b_2:2$.
2. $H:5\times4$, output $5\times2$.
3. Count $3(4)+4+4(2)+2=26$. Biases broadcast across rows.

### 1.3-L1-Q02 — Activation functions and gradients

At z=-2 with incoming gradient 3, compute ReLU and leaky-ReLU outputs and gradients with respect to z.

**Formula support:** $\sigma^\prime=\sigma(1-\sigma)$; $\tanh^\prime=1-\tanh^2$; ReLU slope is 0 below zero and 1 above; leaky ReLU uses negative slope 0.01.

**Step-by-step solution**

1. ReLU output is 0; local derivative 0; incoming gradient becomes $3(0)=0$.
2. Leaky-ReLU output is $0.01(-2)=-0.02$; local derivative 0.01; propagated gradient is 0.03.

### 1.3-L1-Q03 — Piecewise-linear representation

Write f(x)=ReLU(x)-ReLU(x-2) in piecewise form, calculate f(-1), f(1), f(3), and identify slope changes.

**Formula support:** $\operatorname{ReLU}(z)=\max(0,z)$; $f(x)=a_0+a_1x+\sum_jc_j\operatorname{ReLU}(x-b_j)$.

**Step-by-step solution**

1. For $x\le0$, both terms are zero.
2. For $0<x<2$, only the first is active: $f=x$.
3. For $x\ge2$, $f=x-(x-2)=2$.
4. Values are 0, 1, 2; slopes change from 0 to 1 at 0 and from 1 to 0 at 2.
