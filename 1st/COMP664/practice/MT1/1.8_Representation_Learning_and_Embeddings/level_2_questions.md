# 1.8 Representation Learning & Embeddings — Level 2

Sources: [section summary](../../../MT1/1.8%20Representation%20Learning%20%26%20Embeddings/1.8_Representation_Learning_and_Embeddings.md) and [study notes](../../../MT1/1.8%20Representation%20Learning%20%26%20Embeddings/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.8-L2-Q01 — Embedding lookup and context supervision

A vocabulary has 1000 tokens and embedding width 32. Count table parameters. In “a small bird can fly,” list radius-2 skip-gram targets for “bird” and explain how CBOW reverses the task.

**Formula support:** For a one-hot row $e_c$, lookup is $e_cE$ with $E\in\mathbb R^{V\times d}$; radius $r$ gives up to $2r$ context targets.

### 1.8-L2-Q02 — Full-softmax skip-gram

Use scalar u=1, v=(0,ln 3), and observed context token 1. Compute probabilities, loss, and center gradient, then describe its descent direction.

**Formula support:** $p(w|c)=e^{v_w^\top u_c}/\sum_je^{v_j^\top u_c}$; $\ell=-\ln p(o|c)$; $\nabla_{u_c}\ell=\sum_wp(w|c)v_w-v_o$.

### 1.8-L2-Q03 — Negative sampling

One positive score is ln 3 and one negative score is -ln 3. Compute negative-sampling loss and both score derivatives.

**Formula support:** $\ell=-\ln\sigma(v_o^\top u)-\sum_k\ln\sigma(-v_{n_k}^\top u)$; positive-score derivative $\sigma(s)-1$, negative-score derivative $\sigma(s)$.

### 1.8-L2-Q04 — Similarity, transfer, and context

Compute cosine similarity between u=(1,2) and v=(2,4). Compare freezing versus fine-tuning a pretrained embedding table on a small labeled dataset.

**Formula support:** $\cos(u,v)=u^\top v/(\|u\|\|v\|)$; vector analogy candidate $e_b-e_a+e_c$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
