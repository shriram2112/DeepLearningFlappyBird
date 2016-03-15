# Using Deep Q Networks to Learn How To Play Flappy Bird

<img src="./flappy_bird_demp.gif" width="250">

Longer version: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)

## Overview
This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning **[2]** and shows that this learning algorithm can be further generalized to the notorious Flappy Bird.

## Installation Dependencies:
* Python 2.7
* TensorFlow
* pygame
* OpenCV-Python

## How to Run?
1. `git clone git@github.com:yenchenlin1994/DeepLearningFlappyBird.git`
2. `cd DeepLearningFlappyBird`
3. `python deep_q_network.py`

## Background

Reinforcement learning develops control patterns by providing feedback on a model’s selected actions, which encourages the model to select better actions in the future. At each time step, given some state s, the model will select an action s, and then observe the new state s' and a reward r based on some optimality criterion.

We specifically used a method known as Q learning, which approximates the maximum expected return for performing an action at a given state using an action-value (Q) function. Specifically, return gives the sum of the rewards until the game terminates, where the reward is discounted by a factor of γ at each time step. We formally define this as:

![alt-text](http://imgur.com/h7MJxSJ.png "(1)")

We then define the action-value function:

![alt-text](http://imgur.com/05MxGxk.png "(2)")

Note that if the optimal Q function is known for state s', we can write the optimal Q function at preceding state s as the maximum expected value of ![alt-text](http://imgur.com/1RSOCHo.png "Sorry, no alt-text for this one"). This identity is known as the Bellman equation:

![alt-text](http://imgur.com/BERyjr2.png "(3)")

The intuition behind reinforcement learning is to continually update the action-value function based on observations using the Bellman equation. It has been shown by Sutton et al 1998 [2] that such update algorithms will converge on the optimal action-value function as time approaches infinity. Based on this, we can define Q as the output of a neural network, which has weights θ, and train this network by minimizing the following loss function at each iteration i:

![alt-text](http://imgur.com/3gFka35.png "(4)")

Where y_i represents the target function we want to approach during each iteration. It is defined as:

![alt-text](http://imgur.com/gKcXJfi.png "(5)")

Note that when i is equal to the final iteration of an episode (colloquially the end of a game), the Q function should be 0 since it is impossible to attain additional reward after the game has ended. Therefore, when i equals the terminal frame of an episode, we can simply write:

![alt-text](http://imgur.com/nU8qRJM.png "(6)")

## Deep Q Learning Algorithm

The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```

## Experimental Methods

The network was trained on the raw pixel values observed from the game at each time step. We preprocessed the images by converting to grayscale, resizing them to 80x80, and then stacked together the last four frames to produce an 80x80x4 input array.

The architecture of the network is described in Figure 1 below. The first layer convolves the input image with an 8x8x4x32 kernel at a stride size of 4. The output is then put through a 2x2 max pooling layer. The second layer convolves with a 4x4x32x64 kernel at a stride of 2. We then max pool again. The third layer convolves with a 3x3x64x64 kernel at a stride of 1. We then max pool one more time. The last hidden layer consists of 256 fully connected ReLU nodes.

![alt-text](http://imgur.com/mfatQrY.png "Figure 1")

The output layer, obtained with a simple matrix multiplication, has the same dimensionality as the number of valid actions which can be performed in the game, where the 0th index always corresponds to doing nothing. The values at this output layer represent the Q function given the input state for each valid action. At each time step, the network performs whichever action corresponds to the highest Q value using a ϵ greedy policy.

To make it converge faster, I remove the background that appeared in the original game.

At startup, I initialize all weight matrices randomly using a normal distribution with a standard deviation of 0.01, then set the replay memory with a max size of 500,00 experiences.

I start training by choosing actions uniformly at random for 10,000 time steps, without updating the network weights. This allows the ner to populate the replay memory before training begins. After that, I linearly anneal ϵ from 0.1 to 0.001 over the course of the next 1000,000 frames. Why I initialize from 0.1 instead of 1 which is suggested in the original paper because flappy bird is very sensitive to the **flap** action. During this time, at each time step, the network samples minibatches of size 32 from the replay memory to train on, and performs a gradient step on the loss function described above using the Adam optimization algorithm with a learning rate of 0.000001. After annealing finishes, the network continues to train indefinitely, with ϵ fixed at 0.001.


## References

**[1]** Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

**[2]** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

**[3]** Kevin Chen. [Report](http://cs229.stanford.edu/proj2015/362_report.pdf)

## Disclaimer
This work is highly based on [asrivat1's DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)



