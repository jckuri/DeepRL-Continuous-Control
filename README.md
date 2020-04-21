# DeepRL-Navigation
Project 1 "Navigation" of the Deep Reinforcement Learning nanodegree.

## Training Code

You can find the training code here: [Navigation.ipynb](Navigation.ipynb) and [dqn_agent.py](dqn_agent.py)

## Saved Model Weights

You can find the saved model weights here: [checkpoint.pth](checkpoint.pth)

## Project Details

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

<p align="center">
 <img src="/images/bananas.gif">
</p>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

Follow the instructions in this link in order to install all the dependencies required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning#dependencies

Download the `Project 1 - Navigation` into your computer:<br/>
https://github.com/jckuri/DeepRL-Navigation

Follow the instructions in this link in order to install the Unity environment required to run this project:<br/>
https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started

The easiest way to install the requirements is to use the file [requirements.txt](python/requirements.txt)
```
tensorflow==1.7.1
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==0.4.0
pandas
scipy
ipykernel
```

Execute this command in order to install the software specified in `requirements.txt`<br/>
```pip -q install ./python```<br/>
This command is executed at the beginning of the Jupyter notebook [Navigation.ipynb](Navigation.ipynb).

If you have troubles when installing this project, you can write me at:<br/>
https://www.linkedin.com/in/jckuri/

## Instructions

Follow the instructions in [Navigation.ipynb](Navigation.ipynb) to get started with training your own agent!

To run all the cells in the Jupyter notebook again, go to the Jupyter notebook menu, and click on `Kernel` => `Restart & Run All`.

At the end of the Jupyter notebook, there is a space in which you can program your own implementation of this DQN Agent.

## Report

You can find the report here: [Report.md](Report.md)
