# 1.8 Representation Learning & Embeddings — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.8-L2-Q01 — Embedding lookup and context supervision

Table has 32,000 parameters. Targets are a, small, can, fly. Skip-gram predicts those from bird; CBOW predicts bird from the surrounding context. The text supplies the labels.

## 1.8-L2-Q02 — Full-softmax skip-gram

Probabilities $(1/4,3/4)$; loss $\ln4\approx1.3863$; gradient $(3/4)\ln3\approx0.8240$. Descent decreases u, lowering token-2 compatibility relative to the observed token 1.

## 1.8-L2-Q03 — Negative sampling

Positive probability $\sigma(\ln3)=3/4$; negative correctness probability $\sigma(-(-\ln3))=3/4$. Loss $-2\ln(3/4)\approx0.5754$. Score gradients are -1/4 and +1/4.

## 1.8-L2-Q04 — Similarity, transfer, and context

Cosine $10/(\sqrt5\sqrt{20})=1$: vectors differ in magnitude but share direction. Freezing reduces trainable parameters and can preserve pretrained structure; fine-tuning adapts to the task but can overfit or distort useful relations, especially with little data.
