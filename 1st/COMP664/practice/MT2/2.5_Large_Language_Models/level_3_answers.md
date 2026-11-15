# 2.5 Large Language Models — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.5-L3-Q01 — Pretraining, prompting, and model calls

Text supplies its own shifted labels, requiring no external label for every token. Training rewards matching observed continuations, which can contain errors and need not provide evidence for a claim. A single call extends one conditioned sequence; a system can orchestrate several calls and pass outputs between them. The latter is a system workflow, not a change to the basic token predictor.

## 2.5-L3-Q02 — Scaling laws and apparent emergence

Evaluate several scales using comparable datasets and training/evaluation procedures, recording size conventions, training tokens, and FLOPs. Measure both discrete success and smoother probability/loss metrics across many examples. Quantify sampling variation and test whether a sharp transition survives metric changes. Control compute/data tradeoffs; avoid inferring a universal capability law from one benchmark.

## 2.5-L3-Q03 — Long contexts and sparse attention

Sparse methods restrict interacting pairs; approximate/linear methods alter attention computation and may change its result. Exact tiled methods can avoid storing the full score matrix while retaining quadratic arithmetic. Retrieval selects relevant external passages for the input and does not itself alter the attention algorithm. Reduced memory is not automatically reduced asymptotic arithmetic.

## 2.5-L3-Q04 — KV cache and head sharing

Prefill processes a known prompt; decode extends a sequence. Caching reuses earlier K/V rather than recomputing them. GQA/MQA reduce KV heads, latent methods compress stored representations, quantization reduces bytes, paging reduces storage waste, and windows limit retained history. Each ordinary full-attention decode query still reads earlier K/V, so attention work and traffic depend on context length.

## 2.5-L3-Q05 — Bandwidth, quantization, and adapters

Latency is request waiting time; throughput is total tokens per second. Batching amortizes weight reads over sequences but increases state/cache traffic and can affect latency. Quantization uses fewer bits with approximation error. LoRA freezes base weights and trains low-rank additions; QLoRA combines low-bit frozen storage with adapters. Raw size omits scales/metadata, KV cache, activations, optimizer state, and communication; kernel support and accuracy affect realized gains.

## 2.5-L3-Q06 — Speculative decoding

Accepted mass for token x is $q(x)\min(1,p/q)=\min(p,q)$. Rejection probability equals $Z=1-\sum\min(p,q)$; corrected residual adds $(p-q)_+$. Total is p for every token. If Z=0, rejection cannot occur. Greedy verification instead accepts matching argmax choices and replaces the first mismatch. Speed depends on acceptance, draft/verification costs, batching, and overhead; poor acceptance can eliminate gains.

## 2.5-L3-Q07 — Interpretability and ablation

Create varied pronoun-resolution tests plus unrelated controls, measure baseline and targeted/matched-control ablations, and report uncertainty across examples. Attention maps motivate hypotheses but do not establish causation. Ablations give setting-specific intervention evidence and can have nonspecific effects. Trust under shift additionally requires representative stress tests, monitoring, and evidence in the deployment setting; one mechanism study is insufficient.
