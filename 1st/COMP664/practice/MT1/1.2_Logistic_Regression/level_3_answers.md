# 1.2 Logistic Regression — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.2-L3-Q01 — Binary probabilities and training

Independent Bernoulli likelihood is $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$. Negative log gives summed binary cross-entropy. For one example, $\partial\ell/\partial p=(p-y)/[p(1-p)]$. Multiplication by $dp/dz=p(1-p)$ gives $p-y$.

## 1.2-L3-Q02 — Multiclass softmax

The factor $e^a$ cancels between numerator and denominator of softmax. Independent sigmoids need not sum to one and model separate binary outcomes; softmax normalizes a mutually exclusive class distribution.

## 1.2-L3-Q03 — Decision boundaries and averaging

The model produces probabilities from fixed parameters. A threshold maps those probabilities to labels; raising it shrinks the class-1 region. Threshold selection can reflect decision costs and validation goals without retraining the probability model.
