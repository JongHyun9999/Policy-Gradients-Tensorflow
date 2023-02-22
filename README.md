# Policy Gradients

In Reinforcement Learning, we build autonomous agents to complete some task. In this case, we will be training an agent to play the game of CartPole. CartPole is a simple game where a pole is attached to a cart. The goal of the game is to balance the pole on the cart. The cart can move left or right. The game ends when the pole falls over or the cart moves out of bounds. The game is considered solved when the agent can balance the pole for 200 consecutive episodes. 

![CartPole](cartpole.png)

This game is very simple and is not focus of this repo. So, our focus will be construct Agent from scratch using some type algorithm like Policy Gradient and to prove all math that will all of this work. We will use tensorflow to build our agent and Gymnasium (Gym) to provide environment. So, lets hands on!

# Algorithm and math
## 1) Probabilities of trajectory
