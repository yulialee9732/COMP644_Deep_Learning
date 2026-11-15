# 2.5 Large Language Models — Level 3

Sources: [section summary](../../../MT2/2.5%20LLM/2.5_Large_Language_Models.md) and [study notes](../../../MT2/2.5%20LLM/note.md).

Independent questions with no supplied formulas or hints. Answers are in the separate [Level 3 answer sheet](level_3_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.5-L3-Q01

Explain why next-token training is self-supervised and why neither low training loss nor fluent output guarantees factual correctness. Distinguish one call generating many tokens from a system making several calls.

### 2.5-L3-Q02

Design an experiment to test an apparent emergent ability. Explain how model size, tokens, training compute, evaluation metric, and uncertainty should enter the analysis.

### 2.5-L3-Q03

Compare sparse, approximate/linear, and memory-efficient exact attention with retrieval-augmented generation. Distinguish arithmetic complexity from materialized score memory.

### 2.5-L3-Q04

Explain prefill and decode, why KV caching helps, and how GQA, MQA, latent compression, cache quantization, paging, and sliding windows affect memory. Does caching make full-context decode independent of context length?

### 2.5-L3-Q05

Explain the difference between latency and throughput, why batching can help bandwidth-limited decoding, and how quantization, LoRA, and QLoRA change storage or trainable parameters. Include at least two limitations of a raw weight-size estimate.

### 2.5-L3-Q06

Prove distribution preservation for corrected speculative sampling. Contrast with greedy verification and explain why a faster draft model does not guarantee an end-to-end speedup.

### 2.5-L3-Q07

Design an ablation study for a suspected pronoun-resolution head. Distinguish observational attention patterns, causal intervention, and evidence sufficient to trust a system under distribution shift.

Solutions are in the separate [Level 3 answer sheet](level_3_answers.md).
