# 2.2 Sequence Generation, Teacher Forcing, Sampling, and Seq2Seq Models — Level 1

Sources: [section summary](../../../MT2/2.2%20SeqGen%2C%20TeacherForcing%2C%20Sampling%2C%20Seq2Seq%20Models/2.2_Sequence_Generation_and_Seq2Seq.md) and [study notes](../../../MT2/2.2%20SeqGen%2C%20TeacherForcing%2C%20Sampling%2C%20Seq2Seq%20Models/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.2-L1-Q01 — Autoregressive generation and teacher forcing

The target is “red fox EOS” with conditional probabilities 0.5, 0.4, 0.25. Write teacher-forced input/target alignment, sequence probability, and summed negative log likelihood.

**Formula support:** $p(y_{1:T}|x)=\prod_tp(y_t|y_{<t},x)$; $L=-\sum_t\ln p(y_t|y_{<t},x)$.

**Step-by-step solution**

1. Inputs: BOS, red, fox; targets: red, fox, EOS.
2. Probability $0.5(0.4)(0.25)=0.05$.
3. Loss $-\ln0.05=\ln20\approx2.9957$.
4. During training use true previous tokens; during generation use selected tokens.

### 2.2-L1-Q02 — Greedy decoding and sampling

At the first step P(A)=0.6 and P(B)=0.4. Their best second-token conditional probabilities are 0.5 and 0.9. Compare greedy length-two decoding with the highest-probability complete path.

**Formula support:** $p_i(T)=e^{z_i/T}/\sum_je^{z_j/T}$ for $T>0$; top-k retains k highest probabilities, top-p retains a smallest sorted prefix reaching cumulative mass p, then renormalizes.

**Step-by-step solution**

1. Greedy chooses A first, then its best continuation, with joint probability $0.6(0.5)=0.30$.
2. The best B continuation has joint probability $0.4(0.9)=0.36$.
3. Thus the best complete path starts with B, showing local greedy choices need not maximize joint probability.

### 2.2-L1-Q03 — Encoder–decoder attention

Two encoder states are h1=(1,0), h2=(0,2), with attention scores (0,ln 3). Compute weights and context; explain why it can change at the next decoder step.

**Formula support:** $\alpha_i=e^{s_i}/\sum_je^{s_j}$; $c=\sum_i\alpha_ih_i$.

**Step-by-step solution**

1. Exponentials (1,3) give weights (1/4,3/4).
2. Context is $(1/4)(1,0)+(3/4)(0,2)=(0.25,1.5)$.
3. A new decoder state can yield new scores and a different source alignment.
