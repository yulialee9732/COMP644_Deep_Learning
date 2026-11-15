# 1.5 Learning Rates and Regularization — Level 2

Sources: [section summary](../../../MT1/1.5%20Learning%20Rates%20and%20Regularization/1.5_Learning_Rates_and_Regularization.md) and [study notes](../../../MT1/1.5%20Learning%20Rates%20and%20Regularization/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.5-L2-Q01 — Learning-rate stability

For a=5 and w0=2, classify eta=0.1, 0.3, 0.4, and 0.5. Give the first updated weight in each case.

**Formula support:** For $L(w)=\tfrac12aw^2$, $w^+=(1-\eta a)w$; convergence requires $0<\eta a<2$.

### 1.5-L2-Q02 — Momentum, RMSProp, and Adam

Repeat a first Adam step with g=-3, beta1=0.8, beta2=0.9, eta=0.02, epsilon=0, and zero moments. Compare its change with ordinary SGD at the same learning rate.

**Formula support:** $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$; $\hat m=m_t/(1-\beta_1^t)$, $\hat v=v_t/(1-\beta_2^t)$; Adam step $=-\eta\hat m/(\sqrt{\hat v}+\epsilon)$.

### 1.5-L2-Q03 — Regularization and dropout

Given w=-3, data gradient 1, lambda=0.2 and eta=0.5, calculate L2 and L1 updates. Apply inverted dropout with p=0.25 to h=(3,6) and mask (1,0).

**Formula support:** $L_{L2}=L+\frac\lambda2\|w\|^2$, $g_{L2}=g+\lambda w$; $g_{L1}=g+\lambda\operatorname{sign}(w)$ away from zero; $\tilde h=m\odot h/(1-p)$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
