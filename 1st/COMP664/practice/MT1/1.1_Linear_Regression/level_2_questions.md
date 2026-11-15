# 1.1 Linear Regression — Level 2

Sources: [section summary](../../../MT1/1.1%20Linear%20Reg/1.1_Linear_Regression.md) and [study notes](../../../MT1/1.1%20Linear%20Reg/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.1-L2-Q01 — Prediction and design matrices

For x=(0,2,4), w=-0.5, b=3, construct the augmented matrix, give its shape, and compute predictions.

**Formula support:** $\hat y=Xw+b\mathbf1$; augment each input with a constant 1 to absorb the bias.

### 1.1-L2-Q02 — Loss, gradients, and loss scaling

Use X=(1,2) as a two-row, one-column matrix, y=(2,3), w=1, no bias, and eta=0.1. Compute the sum loss, its gradient, and one update. Then redo the gradient and update for half the sum loss.

**Formula support:** $L=\|Xw-y\|^2$, $\nabla L=2X^\top(Xw-y)$; $w^+=w-\eta\nabla L$.

### 1.1-L2-Q03 — Minibatches and epochs

For 240 examples, compare update counts for full-batch, strict SGD, and batch size 30. Explain why batch gradients should be averaged for a mean-loss objective.

**Formula support:** $g_B=|B|^{-1}\sum_{i\in B}g_i$; updates per epoch $=N/B$ when divisible.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
