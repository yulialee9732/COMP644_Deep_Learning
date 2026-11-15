# 1.2 Logistic Regression — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.2-L2-Q01 — Binary probabilities and training

$p=0.5$, loss $\ln2\approx0.6931$, $g_w=-1$, $g_b=0.5$. Updated $w=0.2$, $b=-0.1$; the new logit is -0.5, reducing the class-1 probability as desired.

## 1.2-L2-Q02 — Multiclass softmax

$p=(1/6,1/2,1/3)$, loss $\ln3\approx1.0986$, gradient $(1/6,1/2,-2/3)$. Subtracting the maximum logit leaves probabilities unchanged because a common exponential factor cancels.

## 1.2-L2-Q03 — Decision boundaries and averaging

Weight contributions are -0.4 and -0.4; mean -0.4. Bias contributions -0.2 and 0.4; mean 0.1. Descent increases the weight and decreases the bias for this batch.
