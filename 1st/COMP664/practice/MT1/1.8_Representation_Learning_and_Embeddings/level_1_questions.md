# 1.8 Representation Learning & Embeddings — Level 1

Sources: [section summary](../../../MT1/1.8%20Representation%20Learning%20%26%20Embeddings/1.8_Representation_Learning_and_Embeddings.md) and [study notes](../../../MT1/1.8%20Representation%20Learning%20%26%20Embeddings/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.8-L1-Q01 — Embedding lookup and context supervision

Let E=[[1,0],[0,2],[3,4]]. Look up token 3 using a one-hot vector. For “the quick brown fox jumps,” list radius-1 skip-gram pairs centered at “brown.”

**Formula support:** For a one-hot row $e_c$, lookup is $e_cE$ with $E\in\mathbb R^{V\times d}$; radius $r$ gives up to $2r$ context targets.

**Step-by-step solution**

1. One-hot row is $(0,0,1)$; multiplication selects $(3,4)$.
2. Neighboring tokens are “quick” and “fox.”
3. Pairs are (brown, quick) and (brown, fox); center predicts each context token.

### 1.8-L1-Q02 — Full-softmax skip-gram

Use scalar center u=1 and scalar context embeddings v=(0,ln 2). The observed context is token 2. Compute scores, probabilities, loss, and center gradient.

**Formula support:** $p(w|c)=e^{v_w^\top u_c}/\sum_je^{v_j^\top u_c}$; $\ell=-\ln p(o|c)$; $\nabla_{u_c}\ell=\sum_wp(w|c)v_w-v_o$.

**Step-by-step solution**

1. Scores are $(0,\ln2)$, exponentials (1,2).
2. Probabilities $(1/3,2/3)$.
3. Loss $-\ln(2/3)=\ln(3/2)\approx0.4055$.
4. Gradient $0/3+(2/3)\ln2-\ln2=-(\ln2)/3\approx-0.2310$.

### 1.8-L1-Q03 — Negative sampling

A positive pair and two sampled negative pairs all have score zero. Compute the loss and each score derivative, and explain desired update directions.

**Formula support:** $\ell=-\ln\sigma(v_o^\top u)-\sum_k\ln\sigma(-v_{n_k}^\top u)$; positive-score derivative $\sigma(s)-1$, negative-score derivative $\sigma(s)$.

**Step-by-step solution**

1. Every sigmoid probability is 0.5; each term is $\ln2$.
2. Total $3\ln2\approx2.0794$.
3. Positive derivative is -0.5; each negative derivative is +0.5.
4. Descent raises the positive score and lowers negative scores when considering score coordinates.

### 1.8-L1-Q04 — Similarity, transfer, and context

For u=(1,0), v=(1,1), compute cosine similarity. Explain why a single static embedding for “bank” cannot directly distinguish its financial and river senses.

**Formula support:** $\cos(u,v)=u^\top v/(\|u\|\|v\|)$; vector analogy candidate $e_b-e_a+e_c$.

**Step-by-step solution**

1. Dot product is 1; norms are 1 and $\sqrt2$.
2. Cosine is $1/\sqrt2\approx0.7071$.
3. A static lookup always returns one row for “bank”; context-sensitive downstream processing is needed to distinguish senses.
