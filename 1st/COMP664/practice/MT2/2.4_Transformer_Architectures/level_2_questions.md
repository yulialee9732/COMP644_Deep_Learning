# 2.4 Transformer Architectures — Level 2

Sources: [section summary](../../../MT2/2.4%20Transformer%20Architectures/2.4_Transformer_Architectures.md) and [study notes](../../../MT2/2.4%20Transformer%20Architectures/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.4-L2-Q01 — Transformer blocks and parameter counts

For N=4, d=32, V=200, estimate the same count. How many extra weights would an untied output matrix require, ignoring output biases?

**Formula support:** FFN$(x)=\phi(xW_1+b_1)W_2+b_2$; width $d\to4d\to d$ gives $8d^2$ weights; attention $4d^2$; tied-embedding total $P\approx12Nd^2+Vd$.

### 2.4-L2-Q02 — Positions, embeddings, and logits

A vocabulary has 500 tokens and width 24. Give embedding and tied output-matrix shapes and parameter count. Explain how a word split into three subwords changes occupied positions compared with a one-token word.

**Formula support:** $x_i=e(w_i)+p_i$; for tied embedding matrix $E:V\times d$, output logits for row h are $s=hE^\top$.

### 2.4-L2-Q03 — Causal masks and cross-attention

For four positions with equal raw scores, write the causal matrix. In cross-attention, the decoder has length 3 and encoder length 5; give the score-matrix shape per head.

**Formula support:** Causal mask $M_{ij}=0$ if $j\le i$, otherwise $-\infty$; $A=\mathrm{softmax}_{row}(S+M)$. Cross-attention Q comes from decoder, K/V from encoder.

### 2.4-L2-Q04 — Architecture and objective selection

Explain BERT masked-language modeling, T5 text-to-text task prefixes, and GPT causal next-token training. For a BERT-style masked example with two target probabilities 0.5 and 0.25, calculate the summed masked-token loss.

**Formula support:** Autoregressive loss $=-\sum_t\ln p(y_t|y_{<t},x)$; masked-token loss $=-\sum_{i\in M}\ln p(x_i|\text{corrupted input})$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
