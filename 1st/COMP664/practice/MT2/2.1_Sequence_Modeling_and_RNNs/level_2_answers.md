# 2.1 Sequence Modeling and RNNs — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.1-L2-Q01 — Markov histories and sequence probability

Probability $0.2(0.5)(0.3)=0.03$. Possible contexts number 100 and 10,000, ignoring special boundary tokens. Higher order increases data sparsity.

## 2.1-L2-Q02 — Recurrent forward computation

$h_1=\tanh1\approx0.762$, $h_2=\tanh(0.761594-1)\approx-0.234$. $W_{hh}:5\times5$, $W_{xh}:5\times3$, $b:5$.

## 2.1-L2-Q03 — Backpropagation through time

Products are $0.2^5=0.00032$ and $1.5^5=7.59375$. The former contracts strongly, the latter expands. Matrix RNN behavior depends on products of full Jacobians, not a single scalar coefficient alone.
