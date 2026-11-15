# 1.5 Learning Rates and Regularization — Level 1

Sources: [section summary](../../../MT1/1.5%20Learning%20Rates%20and%20Regularization/1.5_Learning_Rates_and_Regularization.md) and [study notes](../../../MT1/1.5%20Learning%20Rates%20and%20Regularization/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.5-L1-Q01 — Learning-rate stability

Let a=4 and initial w=2. Compare the first step and long-run behavior for eta=0.1 and eta=0.3.

**Formula support:** For $L(w)=\tfrac12aw^2$, $w^+=(1-\eta a)w$; convergence requires $0<\eta a<2$.

**Step-by-step solution**

1. At eta 0.1, multiplier is $1-0.4=0.6$, so first weight is 1.2. Repeated steps converge without sign changes.
2. At eta 0.3, multiplier is $1-1.2=-0.2$, so first weight is -0.4. Repeated steps oscillate while converging.
3. Both multipliers have magnitude below one.

### 1.5-L1-Q02 — Momentum, RMSProp, and Adam

For the first Adam step, let g=2, m0=v0=0, beta1=0.9, beta2=0.99, eta=0.1, and epsilon=0. Calculate both moments, corrected moments, and the parameter change.

**Formula support:** $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$; $\hat m=m_t/(1-\beta_1^t)$, $\hat v=v_t/(1-\beta_2^t)$; Adam step $=-\eta\hat m/(\sqrt{\hat v}+\epsilon)$.

**Step-by-step solution**

1. $m_1=0.1(2)=0.2$.
2. $v_1=0.01(4)=0.04$.
3. $\hat m_1=0.2/0.1=2$, $\hat v_1=0.04/0.01=4$.
4. Change $=-0.1(2)/\sqrt4=-0.1$. The denominator is nonzero in this example.

### 1.5-L1-Q03 — Regularization and dropout

Given w=2, data gradient 3, lambda=0.5 and eta=0.1, calculate L2 and L1 updates. For h=4 and dropout probability 0.5, list possible inverted-dropout outputs and their expectation.

**Formula support:** $L_{L2}=L+\frac\lambda2\|w\|^2$, $g_{L2}=g+\lambda w$; $g_{L1}=g+\lambda\operatorname{sign}(w)$ away from zero; $\tilde h=m\odot h/(1-p)$.

**Step-by-step solution**

1. L2 gradient is $3+0.5(2)=4$, so updated weight 1.6.
2. L1 gradient is $3+0.5=3.5$, so updated weight 1.65.
3. Dropout outputs 0 or $4/0.5=8$, each with probability 0.5.
4. Expectation is 4; inference uses h=4 directly.
