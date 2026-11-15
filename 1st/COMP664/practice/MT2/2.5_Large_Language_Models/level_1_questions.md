# 2.5 Large Language Models — Level 1

Sources: [section summary](../../../MT2/2.5%20LLM/2.5_Large_Language_Models.md) and [study notes](../../../MT2/2.5%20LLM/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.5-L1-Q01 — Pretraining, prompting, and model calls

Three next-token target probabilities are 0.5, 0.25, and 0.8. Calculate summed and mean pretraining loss. Explain whether adding an example to the prompt updates weights.

**Formula support:** $L(\theta)=-\sum_t\ln p_\theta(x_{t+1}|x_{1:t})$; prompting changes context; fine-tuning updates $\theta$.

**Step-by-step solution**

1. Joint target probability is $0.5(0.25)(0.8)=0.1$.
2. Summed loss $=-\ln0.1=\ln10\approx2.3026$.
3. Mean is $\ln10/3\approx0.7675$.
4. An ordinary prompt changes conditioning, not model weights.

### 2.5-L1-Q02 — Scaling laws and apparent emergence

Under an illustrative law with alpha=1/2, how much must model size increase to halve excess loss? Explain the meaning of excess loss and one limit of extrapolation.

**Formula support:** Illustrative law $L(N)=L_\infty+aN^{-\alpha}$, $a,\alpha>0$; excess-loss ratio for size factor c is $c^{-\alpha}$.

**Step-by-step solution**

1. Excess loss is $L-L_\infty=aN^{-1/2}$.
2. Set $c^{-1/2}=1/2$; thus $c=4$.
3. Quadrupling size halves the modeled excess, not necessarily total loss. The fit applies only under its empirical conditions; data or compute limits may break extrapolation.

### 2.5-L1-Q03 — Long contexts and sparse attention

For L=1000, count dense scores and allowed causal pairs. Bound local attention pairs for w=32, and explain why retrieval does not change dense attention complexity for a resulting prompt.

**Formula support:** Dense scores $L^2$; allowed causal pairs $L(L+1)/2$; a local window of at most w keys per query uses at most Lw pairs, with boundary corrections.

**Step-by-step solution**

1. Dense count $1000^2=1,000,000$.
2. Causal count $1000(1001)/2=500,500$.
3. Local bound $1000(32)=32,000$; boundaries can reduce it.
4. Retrieval chooses which passages enter the prompt; ordinary dense attention still scales quadratically in that prompt length.

### 2.5-L1-Q04 — KV cache and head sharing

For 2 layers, 4 KV heads, head width 8, length 100, batch 1, and 2 bytes per scalar, calculate cache bytes. Recalculate with one KV head, keeping query-head count unchanged.

**Formula support:** $M_{KV}=2n_\ell n_{kv}d_hLBb$ bytes; factor 2 counts keys and values. MHA: KV heads equal query heads; GQA shares groups; MQA has one KV head.

**Step-by-step solution**

1. Baseline $2(2)(4)(8)(100)(1)(2)=25,600$ bytes.
2. One KV head gives $2(2)(1)(8)(100)(1)(2)=6400$ bytes.
3. Fourfold reduction follows from sharing KV heads; distinct queries can still produce distinct attention weights.

### 2.5-L1-Q05 — Bandwidth, quantization, and adapters

A 1-billion-weight model uses 16-bit storage and memory bandwidth 100 GB/s. Compute raw storage and idealized weight-read time using decimal units. For a 100×100 layer and LoRA rank 2, count adapter weights.

**Formula support:** Raw weight bytes $=Pq/8$ for q bits; idealized $t\gtrsim\max(M/\mathrm{bandwidth},F/\mathrm{FLOP\ rate})$; LoRA $W=W_0+BA$, $B:d\times r$, $A:r\times k$, adapter count $r(d+k)$.

**Step-by-step solution**

1. Weight bytes $10^9(16/8)=2\times10^9$ = 2 GB.
2. Read time $2/100=0.02$ s = 20 ms, excluding other work.
3. Adapter count $2(100+100)=400$ versus 10,000 base weights, or 4%.
4. The frozen base still participates in computation.

### 2.5-L1-Q06 — Speculative decoding

For target p=(0.6,0.4) and draft q=(0.3,0.7), compute acceptance probabilities, accepted masses, rejection probability, residual distribution, and final token probabilities.

**Formula support:** Accept draft token $x\sim q$ with $a(x)=\min(1,p(x)/q(x))$; rejected-token residual $r(x)=(p(x)-q(x))_+/Z$, $Z=\sum_x(p(x)-q(x))_+$. Accepted mass $=\min(p,q)$.

**Step-by-step solution**

1. Acceptance probabilities are $(1,4/7)$.
2. Accepted masses are $(0.3,0.4)$.
3. Rejection probability $1-0.7=0.3$.
4. Positive residual $(0.3,0)$ normalizes to $(1,0)$.
5. Add residual mass $(0.3,0)$ to get $(0.6,0.4)=p$.

### 2.5-L1-Q07 — Interpretability and ablation

A model scores 90% before a head ablation and 84% after. Calculate percentage-point and relative drops. Explain one reason this does not prove the head exclusively performs the tested skill.

**Formula support:** Effect estimate $\Delta=m_{baseline}-m_{ablated}$; relative drop $=\Delta/m_{baseline}$. Interpret with matched controls and sampling uncertainty.

**Step-by-step solution**

1. Absolute drop $90-84=6$ percentage points.
2. Relative drop $6/90\approx6.67\%$.
3. Removing the head could disrupt general information flow or interact with other components, so the effect alone does not establish an exclusive semantic function.
