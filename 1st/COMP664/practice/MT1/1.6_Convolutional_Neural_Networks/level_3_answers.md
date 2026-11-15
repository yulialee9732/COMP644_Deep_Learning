# 1.6 Convolutional Neural Networks — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.6-L3-Q01 — Convolution shapes and parameters

Each filter has $K_hK_wC_{in}$ weights and one optional bias. Multiply by $C_{out}$. Reusing the same filter at every location prevents spatial size from increasing this count. Dense layers learn a separate weight for each input-output pair; convolution restricts receptive fields and ties location-specific weights.

## 1.6-L3-Q02 — Pooling and shared-weight gradients

Convolution weight gradients sum contributions over all locations and examples. Max pooling routes a window gradient to the selected maximum under a tie convention; average pooling distributes it equally. When windows overlap, each input accumulates contributions from every window that used it.

## 1.6-L3-Q03 — Receptive fields and architecture

Equivariance means a shifted input shifts corresponding features; invariance means output remains unchanged. Convolution approximately preserves shifts subject to padding/stride boundaries. Pooling reduces resolution and sensitivity to small shifts but does not guarantee full invariance. Stacked local operations grow receptive fields; a classifier uses the resulting features. LeNet illustrates character recognition; AlexNet combines convolution, pooling, ReLU, dropout, and fully connected classification with GPU training.
