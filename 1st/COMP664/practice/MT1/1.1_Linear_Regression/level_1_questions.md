# 1.1 Linear Regression — Level 1

Sources: [section summary](../../../MT1/1.1%20Linear%20Reg/1.1_Linear_Regression.md) and [study notes](../../../MT1/1.1%20Linear%20Reg/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.1-L1-Q01 — Prediction and design matrices

For two examples x=1 and x=3, take w=2 and b=-1. Construct an augmented design matrix and calculate both predictions.

**Formula support:** $\hat y=Xw+b\mathbf1$; augment each input with a constant 1 to absorb the bias.

**Step-by-step solution**

1. Add a constant column: $X_a=[[1,1],[3,1]]$.
2. Stack parameters as $w_a=(2,-1)^\top$.
3. Multiply: $X_aw_a=(1,5)^\top$. The constant column supplies the intercept.

### 1.1-L1-Q02 — Loss, gradients, and loss scaling

One example has x=2, y=4, w=1, no bias, and learning rate 0.1. Compute squared loss, gradient, updated weight, and new loss.

**Formula support:** $L=\|Xw-y\|^2$, $\nabla L=2X^\top(Xw-y)$; $w^+=w-\eta\nabla L$.

**Step-by-step solution**

1. Predict $2(1)=2$; residual $r=2-4=-2$.
2. $L=r^2=4$.
3. $g=2rx=2(-2)(2)=-8$.
4. $w^+=1-0.1(-8)=1.8$.
5. New prediction is 3.6, so new loss is $(-0.4)^2=0.16$.

### 1.1-L1-Q03 — Minibatches and epochs

A dataset has 120 examples and batch size 20. A minibatch contains per-example scalar gradients 2 and 6 in a separate two-example demonstration. Compute the update count and the demonstration mean gradient.

**Formula support:** $g_B=|B|^{-1}\sum_{i\in B}g_i$; updates per epoch $=N/B$ when divisible.

**Step-by-step solution**

1. $120/20=6$ updates per epoch.
2. For the separate two-example minibatch, $g_B=(2+6)/2=4$.
3. Averaging estimates a mean-loss gradient; summing would instead give 8.
