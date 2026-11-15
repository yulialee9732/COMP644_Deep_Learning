# 2.3 Attention & Transformers — Level 2

Sources: [section summary](../../../MT2/2.3%20Attentions%20%26%20Transformers/2.3_Attention_and_Transformers.md) and [study notes](../../../MT2/2.3%20Attentions%20%26%20Transformers/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.3-L2-Q01 — Scaled dot-product attention

Let q=(2,0), k1=(0,1), k2=(ln 3/sqrt(2),0), v1=(1,0), v2=(0,4), and dk=2. Compute scaled scores, weights, and output.

**Formula support:** $Q=XW_Q$, $K=XW_K$, $V=XW_V$; $A=\mathrm{softmax}_{row}(QK^\top/\sqrt{d_k})$, $Z=AV$.

### 2.3-L2-Q02 — Multi-head dimensions and projections

For d=192 and H=6, compute head width, each per-head Q/K/V matrix shape, concatenated width, W_O shape, and total projection weights. What changes if H=12 at fixed d?

**Formula support:** $\mathrm{MHA}(X)=\mathrm{Concat}(Z_1,\ldots,Z_H)W_O$; usually $d_k=d_v=d/H$; standard Q,K,V,O projection weights total $4d^2$.

### 2.3-L2-Q03 — Information paths and attention complexity

Compare sequence lengths 100 and 400 for explicitly materialized full-attention scores with eight heads and batch size one. Compute entries for both and the increase factor.

**Formula support:** Full attention has $L^2$ ordered query–key pairs; explicit scores use $O(L^2)$ entries per head. An RNN first-to-last path uses $L-1$ recurrent transitions.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
