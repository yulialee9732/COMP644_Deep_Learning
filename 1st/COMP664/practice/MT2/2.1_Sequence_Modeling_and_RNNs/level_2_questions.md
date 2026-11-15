# 2.1 Sequence Modeling and RNNs — Level 2

Sources: [section summary](../../../MT2/2.1%20RNN/2.1_Sequence_Modeling_and_RNNs.md) and [study notes](../../../MT2/2.1%20RNN/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.1-L2-Q01 — Markov histories and sequence probability

A second-order model gives P(A)=0.2, P(B|A)=0.5, P(C|A,B)=0.3. Compute the sequence probability. For vocabulary size 100, compare maximum numbers of length-1 and length-2 contexts.

**Formula support:** Order-$k$ Markov: $p(x_t|x_{<t})\approx p(x_t|x_{t-k:t-1})$; $p(x_{1:T})=\prod_tp(x_t|x_{<t})$.

### 2.1-L2-Q02 — Recurrent forward computation

For $h_t=\tanh(h_{t-1}+x_t)$, with $h_0=0$ and inputs $(x_1,x_2)=(1,-1)$, compute $h_1$ and $h_2$ to three decimals. Give parameter-matrix shapes for input width 3 and hidden width 5 in column-vector notation.

**Formula support:** $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$; a linear recurrence can be used as a simplified exercise.

### 2.1-L2-Q03 — Backpropagation through time

Compare gradient propagation over five steps with local derivatives 0.2 and 1.5, assuming final gradient 1. Explain the difference.

**Formula support:** $\partial h_T/\partial h_t=J_TJ_{T-1}\cdots J_{t+1}$; scalar constant Jacobian $a$ gives $a^{T-t}$; shared-weight gradients sum over time.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
