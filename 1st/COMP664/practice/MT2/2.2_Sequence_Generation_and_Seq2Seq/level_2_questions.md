# 2.2 Sequence Generation, Teacher Forcing, Sampling, and Seq2Seq Models — Level 2

Sources: [section summary](../../../MT2/2.2%20SeqGen%2C%20TeacherForcing%2C%20Sampling%2C%20Seq2Seq%20Models/2.2_Sequence_Generation_and_Seq2Seq.md) and [study notes](../../../MT2/2.2%20SeqGen%2C%20TeacherForcing%2C%20Sampling%2C%20Seq2Seq%20Models/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.2-L2-Q01 — Autoregressive generation and teacher forcing

The target is “a blue bird EOS,” with conditional probabilities 0.8, 0.5, 0.25, 0.5. Give teacher-forced alignment, joint probability, summed loss, and mean token loss.

**Formula support:** $p(y_{1:T}|x)=\prod_tp(y_t|y_{<t},x)$; $L=-\sum_t\ln p(y_t|y_{<t},x)$.

### 2.2-L2-Q02 — Greedy decoding and sampling

For probabilities (0.5,0.3,0.15,0.05), compute the distributions after top-k with k=2 and top-p with p=0.9. For logits (ln 4,0), compute temperature-2 probabilities.

**Formula support:** $p_i(T)=e^{z_i/T}/\sum_je^{z_j/T}$ for $T>0$; top-k retains k highest probabilities, top-p retains a smallest sorted prefix reaching cumulative mass p, then renormalizes.

### 2.2-L2-Q03 — Encoder–decoder attention

Three source states are (2,0), (0,2), (2,2), with scores (ln 2,0,0). Compute weights and context, then describe the basic encoder–decoder bottleneck that attention addresses.

**Formula support:** $\alpha_i=e^{s_i}/\sum_je^{s_j}$; $c=\sum_i\alpha_ih_i$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
