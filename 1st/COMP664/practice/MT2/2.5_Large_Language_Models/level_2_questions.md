# 2.5 Large Language Models — Level 2

Sources: [section summary](../../../MT2/2.5%20LLM/2.5_Large_Language_Models.md) and [study notes](../../../MT2/2.5%20LLM/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.5-L2-Q01 — Pretraining, prompting, and model calls

Two target probabilities are 0.2 and 0.5. Calculate summed and mean loss. Compare a prompt containing examples with supervised fine-tuning, and explain a plan–draft–critique multi-call workflow.

**Formula support:** $L(\theta)=-\sum_t\ln p_\theta(x_{t+1}|x_{1:t})$; prompting changes context; fine-tuning updates $\theta$.

### 2.5-L2-Q02 — Scaling laws and apparent emergence

For alpha=1/3, find the model-size multiplier needed to halve excess loss. Explain why a thresholded benchmark can improve abruptly even if loss improves smoothly.

**Formula support:** Illustrative law $L(N)=L_\infty+aN^{-\alpha}$, $a,\alpha>0$; excess-loss ratio for size factor c is $c^{-\alpha}$.

### 2.5-L2-Q03 — Long contexts and sparse attention

For L=2000, compute dense scores, causal pairs, and a local-window upper bound at w=64. If dense scores are explicitly stored as two-byte numbers for one head, compute their decimal MB.

**Formula support:** Dense scores $L^2$; allowed causal pairs $L(L+1)/2$; a local window of at most w keys per query uses at most Lw pairs, with boundary corrections.

### 2.5-L2-Q04 — KV cache and head sharing

For 4 layers, 8 KV heads, width 16, length 256, batch 2, and 2 bytes per scalar, compute cache bytes and MiB. Recalculate for 2 KV heads. Explain what paging and sliding windows change.

**Formula support:** $M_{KV}=2n_\ell n_{kv}d_hLBb$ bytes; factor 2 counts keys and values. MHA: KV heads equal query heads; GQA shares groups; MQA has one KV head.

### 2.5-L2-Q05 — Bandwidth, quantization, and adapters

A 3-billion-weight model uses 4-bit storage and bandwidth 300 GB/s. Compute raw GB and idealized weight-read time. For d=512, k=256, r=8, count LoRA parameters and their percentage of base weights. State what QLoRA changes.

**Formula support:** Raw weight bytes $=Pq/8$ for q bits; idealized $t\gtrsim\max(M/\mathrm{bandwidth},F/\mathrm{FLOP\ rate})$; LoRA $W=W_0+BA$, $B:d\times r$, $A:r\times k$, adapter count $r(d+k)$.

### 2.5-L2-Q06 — Speculative decoding

For p=(0.2,0.8), q=(0.5,0.5), compute acceptance probabilities, accepted masses, rejection probability, residual distribution, and corrected output distribution. Explain why later drafts are discarded after rejection.

**Formula support:** Accept draft token $x\sim q$ with $a(x)=\min(1,p(x)/q(x))$; rejected-token residual $r(x)=(p(x)-q(x))_+/Z$, $Z=\sum_x(p(x)-q(x))_+$. Accepted mass $=\min(p,q)$.

### 2.5-L2-Q07 — Interpretability and ablation

A model scores 80% before and 76% after an ablation. Compute percentage-point and relative drops, then propose two controls and explain why a zero effect would not prove irrelevance.

**Formula support:** Effect estimate $\Delta=m_{baseline}-m_{ablated}$; relative drop $=\Delta/m_{baseline}$. Interpret with matched controls and sampling uncertainty.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
