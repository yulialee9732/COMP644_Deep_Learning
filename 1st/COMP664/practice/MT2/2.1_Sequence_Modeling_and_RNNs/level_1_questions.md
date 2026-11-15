# 2.1 Sequence Modeling and RNNs — Level 1

Sources: [section summary](../../../MT2/2.1%20RNN/2.1_Sequence_Modeling_and_RNNs.md) and [study notes](../../../MT2/2.1%20RNN/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 2.1-L1-Q01 — Markov histories and sequence probability

A first-order model gives P(A)=0.5, P(B|A)=0.4, and P(C|B)=0.25. Compute P(A,B,C), and identify the context available for its final prediction.

**Formula support:** Order-$k$ Markov: $p(x_t|x_{<t})\approx p(x_t|x_{t-k:t-1})$; $p(x_{1:T})=\prod_tp(x_t|x_{<t})$.

**Step-by-step solution**

1. Multiply the start and transition probabilities: $0.5(0.4)(0.25)=0.05$.
2. The final transition uses only B under the first-order assumption.
3. Earlier A is ignored once B is known by this model.

### 2.1-L1-Q02 — Recurrent forward computation

For a simplified linear recurrence $h_t=0.5h_{t-1}+x_t$, with initial state $h_0=0$ and inputs $(x_1,x_2)=(2,1)$, compute $h_1$ and $h_2$. Explain parameter sharing.

**Formula support:** $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$; a linear recurrence can be used as a simplified exercise.

**Step-by-step solution**

1. $h_1=0.5(0)+2=2$.
2. $h_2=0.5(2)+1=2$.
3. The same recurrent coefficient 0.5 is used at each time; it is not a separate parameter per position. This example uses identity activation, not tanh.

### 2.1-L1-Q03 — Backpropagation through time

A scalar recurrence has local state derivative 0.5 at every step. Find the derivative from h0 to h4 and the gradient at h0 if the gradient at h4 is 2.

**Formula support:** $\partial h_T/\partial h_t=J_TJ_{T-1}\cdots J_{t+1}$; scalar constant Jacobian $a$ gives $a^{T-t}$; shared-weight gradients sum over time.

**Step-by-step solution**

1. **Identify the states and local derivatives.** The forward computation follows four transitions:

   $$
   h_0\longrightarrow h_1\longrightarrow h_2\longrightarrow h_3\longrightarrow h_4.
   $$

   At each transition, $\partial h_t/\partial h_{t-1}=0.5$. This means a small change in the previous state produces approximately half as much change in the next state. It does **not** mean that the state value itself must halve: the recurrence could also contain inputs or a bias. Because each state is a scalar, each local Jacobian is just one number.

2. **Apply the chain rule across all four transitions.** The requested derivative measures how $h_4$ changes when $h_0$ changes:

   $$
   \frac{\partial h_4}{\partial h_0}
   =\frac{\partial h_4}{\partial h_3}
   \frac{\partial h_3}{\partial h_2}
   \frac{\partial h_2}{\partial h_1}
   \frac{\partial h_1}{\partial h_0}
   =(0.5)^4
   =\frac{1}{16}
   =0.0625.
   $$

   There are **four factors**, because there are four transitions between $h_0$ and $h_4$, even though there are five states. For example, a small perturbation $\Delta h_0=0.01$ produces $\Delta h_4\approx0.0625(0.01)=0.000625$ along this path.

3. **Distinguish the state derivative from the loss gradient.** Let $L$ be the scalar loss. The given “gradient at $h_4$” means $\partial L/\partial h_4=2$. This measures how the **loss** changes with $h_4$, whereas $\partial h_4/\partial h_0$ measures how the **final state** changes with $h_0$. Combine them using the chain rule:

   $$
   \frac{\partial L}{\partial h_0}
   =\frac{\partial L}{\partial h_4}
   \frac{\partial h_4}{\partial h_0}
   =2\left(\frac{1}{16}\right)
   =\boxed{0.125}.
   $$

   This assumes the gradient reaches $h_0$ through this state chain, with no additional loss branches contributing at intermediate states.

4. **Check by propagating the gradient backward one step at a time.** Each backward step multiplies the incoming loss gradient by the local derivative $0.5$:

   | State | Loss gradient | Calculation |
   | --- | --- | --- |
   | $h_4$ | $2$ | Given |
   | $h_3$ | $1$ | $2(0.5)$ |
   | $h_2$ | $0.5$ | $1(0.5)$ |
   | $h_1$ | $0.25$ | $0.5(0.5)$ |
   | $h_0$ | $0.125$ | $0.25(0.5)$ |

5. **Interpret the result.** The gradient at $h_0$ is only $1/16$ of the gradient at $h_4$. With more transitions, the factor $0.5^{T-t}$ becomes even smaller. This illustrates **vanishing gradients**: a later loss provides a weaker learning signal to earlier states, making long-range dependencies harder to learn.

   The answer is a gradient with respect to a **state**, not a recurrent weight. To find a shared weight's gradient, backpropagation must add the contributions from every time step where that weight is used; the local state derivative alone does not provide enough information to compute it.
