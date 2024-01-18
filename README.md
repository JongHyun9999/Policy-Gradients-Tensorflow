# Policy Gradient Implementation in Reinforcement Learning Repository

This repository features code implementing the policy gradient method in the field of reinforcement learning. 
Experiments were conducted using the cartpole library from gym, and to facilitate foundational understanding, 
only matrix operations supported by TensorFlow and the GradientTape feature were used, without resorting to Keras.

#### Characteristics of Q-Learning for Comparison
Q-Learning technique stores the value of each state in a table and sequentially decides on the action with the highest cumulative value based on this table. 
However, Q-Learning has fundamental issues due to its approach to managing all states in a table:
 - It's challenging to form a table when the number of states is vast or nearly infinite.
 - Similarly, the characteristic of Q-Learning, which requires visiting all states, becomes practically impossible when the number of states is excessively high.

#### Features of Policy Gradient

Policy Gradient differs from Q-Learning. Q-Learning uses a "value-action function" that evaluates the value of actions taken in each state, 
essentially an evaluation function for actions taken by the agent.

In contrast, Policy Gradient involves creating a kind of action manual (policy) that instructs which action to take in a given state. Typically, 
the policy π of Policy Gradient is a conditional probability of selecting an action a∈A given a state s∈S.

The primary difference between Q-Learning and Policy Gradient is the shift from deciding actions based on a Q-Table to using a neural network policy that functions as a formula.

Detailed explanations are provided inside the ipynb files. Thank you.
