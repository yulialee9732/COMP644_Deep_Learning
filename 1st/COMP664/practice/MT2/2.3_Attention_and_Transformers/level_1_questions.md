# 2.3 Attention & Transformers — Level 1

Sources: [section summary](../../../MT2/2.3%20Attentions%20%26%20Transformers/2.3_Attention_and_Transformers.md) and [study notes](../../../MT2/2.3%20Attentions%20%26%20Transformers/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.3-L1-Q01 — Scaled dot-product attention

For one query q=1, keys (0,ln 3), values (2,6), and dk=1, calculate scores, weights, and attention output.

**Formula support:** $Q=XW_Q$, $K=XW_K$, $V=XW_V$; $A=\mathrm{softmax}_{row}(QK^\top/\sqrt{d_k})$, $Z=AV$.

**Step-by-step solution**

1. Scaled scores are $(0,\ln3)$.
2. Softmax weights are $(1/4,3/4)$.
3. Output $=(1/4)2+(3/4)6=5$. The values carry the mixed information; keys determine the matching scores.

### 2.3-L1-Q02 — Multi-head dimensions and projections

For d=128 and H=4, compute each head width, concatenated width, output-projection shape, and projection-weight count, ignoring biases.

**Formula support:** $\mathrm{MHA}(X)=\mathrm{Concat}(Z_1,\ldots,Z_H)W_O$; usually $d_k=d_v=d/H$; standard Q,K,V,O projection weights total $4d^2$.

**Step-by-step solution**

1. Head width $128/4=32$.
2. Concatenation width $4(32)=128$.
3. $W_O$ is $128\times128$.
4. Q,K,V,O each contribute $128^2$ weights in total across heads, giving $4(128^2)=65,536$.

### 2.3-L1-Q03 — Information paths and attention complexity

For L=50 and one attention head, count full-attention score entries and compare the first-to-last dependency path with an RNN.

**Formula support:** Full attention has $L^2$ ordered query–key pairs; explicit scores use $O(L^2)$ entries per head. An RNN first-to-last path uses $L-1$ recurrent transitions.

**Step-by-step solution**

1. Full scores contain $50^2=2500$ entries.
2. An RNN carries the first state through 49 transitions to the last.
3. Full attention can connect the last query directly to the first key/value within one layer, while allowing matrix-parallel score computation.
