# 2.5 Large Language Models — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 2.5-L2-Q01 — Pretraining, prompting, and model calls

Summed loss $\ln10\approx2.3026$ and mean $\ln10/2\approx1.1513$. Prompt examples condition a fixed model; fine-tuning optimizes parameters. In a multi-call workflow, a generated plan, draft, or critique becomes input context for a later call.

## 2.5-L2-Q02 — Scaling laws and apparent emergence

$c^{-1/3}=1/2$ gives $c=8$. A discrete scoring threshold can turn smooth increases in underlying success probabilities into a sharp score change; apparent emergence depends on the metric and task.

## 2.5-L2-Q03 — Long contexts and sparse attention

Dense 4,000,000; causal 2,001,000; local upper bound 128,000. Explicit dense storage is 8,000,000 bytes = 8 MB. This is only a score tensor, excluding other model state.

## 2.5-L2-Q04 — KV cache and head sharing

Baseline $2(4)(8)(16)(256)(2)(2)=1,048,576$ bytes = 1 MiB. Two KV heads use 262,144 bytes = 0.25 MiB. Paging reduces allocation waste/fragmentation; a sliding window reduces retained history and direct access to distant tokens.

## 2.5-L2-Q05 — Bandwidth, quantization, and adapters

Raw bytes $3\times10^9/2=1.5$ GB; read time $1.5/300=0.005$ s = 5 ms. Adapter count $8(512+256)=6144$; base count 131,072; ratio 4.6875%. QLoRA stores the frozen base in a low-bit format while training adapters at higher precision. Metadata, activations, cache, and optimizer memory are additional.

## 2.5-L2-Q06 — Speculative decoding

Acceptance $(0.4,1)$; accepted mass $(0.2,0.5)$; rejection 0.3; residual $(0,1)$; corrected total $(0.2,0.8)$. Later proposals were conditioned on the rejected prefix, so they are no longer valid proposals for the corrected prefix.

## 2.5-L2-Q07 — Interpretability and ablation

Drop 4 percentage points, relative drop 5%. Compare matched random-head ablations and unrelated tasks, ideally with repeated examples/seeds and uncertainty estimates. A zero effect can reflect redundancy or insensitive evaluation rather than irrelevance.
