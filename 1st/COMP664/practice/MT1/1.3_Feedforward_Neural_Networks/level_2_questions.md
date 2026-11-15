# 1.3 Feedforward Neural Networks — Level 2

Sources: [section summary](../../../MT1/1.3%20FNN/1.3_Feedforward_Neural_Networks.md) and [study notes](../../../MT1/1.3%20FNN/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.3-L2-Q01 — Network dimensions and parameters

A network has dimensions 5→6→3 and batch size 10. Give all parameter and activation shapes, count parameters, and choose an output activation for mutually exclusive classes.

**Formula support:** $H=\phi(XW_1+b_1)$, $Y=g(HW_2+b_2)$; parameters $=dh+h+hC+C$.

### 1.3-L2-Q02 — Activation functions and gradients

At z=2 with incoming gradient -4, compute ReLU and leaky-ReLU outputs and input gradients. Then compare sigmoid derivatives at sigmoid(z)=0.5 and 0.99.

**Formula support:** $\sigma^\prime=\sigma(1-\sigma)$; $\tanh^\prime=1-\tanh^2$; ReLU slope is 0 below zero and 1 above; leaky ReLU uses negative slope 0.01.

### 1.3-L2-Q03 — Piecewise-linear representation

Write g(x)=2ReLU(x-1)-ReLU(x-3) in piecewise form and evaluate g(0), g(2), and g(4).

**Formula support:** $\operatorname{ReLU}(z)=\max(0,z)$; $f(x)=a_0+a_1x+\sum_jc_j\operatorname{ReLU}(x-b_j)$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
