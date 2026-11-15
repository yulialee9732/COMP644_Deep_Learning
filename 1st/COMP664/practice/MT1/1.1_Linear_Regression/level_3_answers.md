# 1.1 Linear Regression — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.1-L3-Q01 — Prediction and design matrices

Explicit bias: $X:100\times5$, $w:5$, $b$ scalar, predictions $100$. Augmented: $X_a:100\times6$, $w_a:6$, predictions $100$. The added column is all ones.

## 1.1-L3-Q02 — Loss, gradients, and loss scaling

For residual $r_i=x_i^\top w-y_i$, $\partial r_i^2/\partial w_j=2r_ix_{ij}$; sum to obtain $2X^\top(Xw-y)$. Hessian is $2X^\top X$. Its quadratic form is $2\|Xv\|^2\ge0$, proving convexity. Full column rank makes it positive definite and the minimizer unique.

## 1.1-L3-Q03 — Minibatches and epochs

Full-batch uses all examples per update and few updates per epoch; strict SGD uses one and has noisier updates; minibatches combine vectorization with more frequent updates. An epoch is a full data pass under ordinary sampling without replacement. More noise does not guarantee better generalization.
