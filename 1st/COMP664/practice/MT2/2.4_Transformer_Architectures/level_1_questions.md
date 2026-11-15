# 2.4 Transformer Architectures — Level 1

Sources: [section summary](../../../MT2/2.4%20Transformer%20Architectures/2.4_Transformer_Architectures.md) and [study notes](../../../MT2/2.4%20Transformer%20Architectures/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.4-L1-Q01 — Transformer blocks and parameter counts

For N=2 blocks, d=16, V=100, and tied input/output embeddings, estimate weights excluding biases, norms, and positions. Show attention, FFN, and vocabulary terms.

**Formula support:** FFN$(x)=\phi(xW_1+b_1)W_2+b_2$; width $d\to4d\to d$ gives $8d^2$ weights; attention $4d^2$; tied-embedding total $P\approx12Nd^2+Vd$.

**Step-by-step solution**

1. Attention total $2(4)(16^2)=2048$.
2. FFN total $2(8)(16^2)=4096$.
3. Vocabulary term $100(16)=1600$.
4. Sum is 7744 weights.

### 2.4-L1-Q02 — Positions, embeddings, and logits

Use token embedding e=(1,2), positions p1=(0,1) and p2=(2,0), and tied E=[[1,0],[0,1],[1,1]]. Compute both positioned vectors and logits for final hidden row h=(2,1).

**Formula support:** $x_i=e(w_i)+p_i$; for tied embedding matrix $E:V\times d$, output logits for row h are $s=hE^\top$.

**Step-by-step solution**

1. Positioned vectors are $(1,3)$ and $(3,2)$.
2. Dot h with each embedding row: logits $(2,1,3)$.
3. The output has one logit per vocabulary token; tying reuses the input table.

### 2.4-L1-Q03 — Causal masks and cross-attention

For three positions with all raw scores zero, write the causal attention matrix and explain the final-row weights.

**Formula support:** Causal mask $M_{ij}=0$ if $j\le i$, otherwise $-\infty$; $A=\mathrm{softmax}_{row}(S+M)$. Cross-attention Q comes from decoder, K/V from encoder.

**Step-by-step solution**

1. Row 1 permits only column 1: $(1,0,0)$.
2. Row 2 permits columns 1–2: $(1/2,1/2,0)$.
3. Row 3 permits all three: $(1/3,1/3,1/3)$.
4. Equal permitted scores give uniform weights over the available prefix.

### 2.4-L1-Q04 — Architecture and objective selection

Choose encoder-only, encoder–decoder, or decoder-only for bidirectional sentiment features, source-conditioned translation, and open-ended continuation. Explain the choice and objective.

**Formula support:** Autoregressive loss $=-\sum_t\ln p(y_t|y_{<t},x)$; masked-token loss $=-\sum_{i\in M}\ln p(x_i|\text{corrupted input})$.

**Step-by-step solution**

1. Encoder-only supports bidirectional input features; a classifier can be fine-tuned on them.
2. Encoder–decoder encodes the source and generates a target with causal self-attention and cross-attention.
3. Decoder-only predicts successive continuation tokens from a causal prefix. These are suitable choices, not exclusive capabilities.
