# 1.5 Learning Rates and Regularization — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.5-L3-Q01 — Learning-rate stability

$L(\theta+\Delta)=L(\theta)+g^\top\Delta+o(\|\Delta\|)$. With $\Delta=-\eta g$, the first-order change is $-\eta\|g\|^2<0$ for nonzero g. At small eta it dominates the remainder; large steps can make curvature terms dominate and overshoot.

## 1.5-L3-Q02 — Momentum, RMSProp, and Adam

Momentum averages gradients; RMSProp averages squared gradients and divides the current gradient by their root plus epsilon; Adam combines both. First moment tracks direction, second tracks uncentered magnitude. Zero initialization biases early moving averages toward zero, motivating correction. Epsilon protects denominators. Variance subtracts the squared mean, whereas the raw second moment does not.

## 1.5-L3-Q03 — Regularization and dropout

L1 penalizes absolute magnitudes and promotes sparsity; L2 penalizes squared magnitudes and shrinks weights smoothly. Dropout trains with randomly removed features, reducing dependence on individual units. If $m\sim Bernoulli(1-p)$, $E[mh/(1-p)]=h$. At inference use all activations without further rescaling under this convention.
