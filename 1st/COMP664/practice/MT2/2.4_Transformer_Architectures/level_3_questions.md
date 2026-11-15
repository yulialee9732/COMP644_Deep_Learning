# 2.4 Transformer Architectures — Level 3

Sources: [section summary](../../../MT2/2.4%20Transformer%20Architectures/2.4_Transformer_Architectures.md) and [study notes](../../../MT2/2.4%20Transformer%20Architectures/note.md).

Independent questions with no supplied formulas or hints. Answers are in the separate [Level 3 answer sheet](level_3_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.4-L3-Q01

Explain how attention and position-wise feed-forward sublayers divide the work of a Transformer block. Derive their approximate parameter counts and state which weights are shared across positions versus blocks.

### 2.4-L3-Q02

Explain why position-free full self-attention is permutation equivariant and why this is a limitation for language. Describe two sources of order sensitivity in Transformer systems.

### 2.4-L3-Q03

Explain how shifted targets and a causal mask permit parallel next-token training without future-token leakage. Contrast which sequence supplies queries, keys, and values in decoder self-attention and encoder–decoder cross-attention.

### 2.4-L3-Q04

Compare encoder-only, encoder–decoder, and decoder-only designs for classification and conditional generation. Include attention visibility, pretraining objectives, and how an encoder–decoder can produce a text classification label.

Solutions are in the separate [Level 3 answer sheet](level_3_answers.md).
