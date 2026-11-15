# 1.6 Convolutional Neural Networks — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.6-L2-Q01 — Convolution shapes and parameters

Spatial size $\lfloor(28+2-5)/2\rfloor+1=13$. Output $13\times13\times16$. Count $5(5)(3)(16)+16=1216$. Only complete valid kernel placements are included, giving the floor.

## 1.6-L2-Q02 — Pooling and shared-weight gradients

Max output 5 and gradient [[-4,0],[0,0]]. Average output 2 and all input gradients -1. Shared-weight gradient is $4(2)+(-1)(3)=5$, adding both uses.

## 1.6-L2-Q03 — Receptive fields and architecture

After convolution: $(r,j)=(3,1)$. After pooling: $(4,2)$. After final convolution: $(8,2)$. Adjacent final units are centered two input positions apart and each sees an eight-position width.
