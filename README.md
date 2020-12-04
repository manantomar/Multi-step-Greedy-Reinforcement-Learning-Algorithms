# Multi-step Greedy Reinforcement Learning Algorithms

This repository contains the code for mulit-step greedy reinforcement learning algorithms. It mainly includes two variants, discrete action case (DQN) and continous action case (TRPO), based on the paper [Multi-step Greedy Reinforcement Learning Algorithms](https://arxiv.org/abs/1910.02919), which was recently presented at ICML 2020.

This implementation makes use of [Tensorflow](https://github.com/tensorflow/tensorflow) and builds over the code provided by [stable-baselines](https://github.com/hill-a/stable-baselines).

# Getting Started

## Prerequisites

All dependencies are provided in a python virtual-env `requirements.txt` file. Majorly, you would need to install `stable-baselines`, `tensorflow`, and `mujoco_py`.

## Installation

1. Install stable-baselines
~~~
pip install stable-baselines[mpi]==2.7.0
~~~

2. [Download](https://www.roboti.us/index.html) and copy MuJoCo library and license files into a `.mujoco/` directory. We use `mujoco200` for this project.

3. Clone this repository and copy the `deepq_kpi`, `deepq_kvi`, `trpo_kpi`, and `trpo-kvi` directories inside [this directory](https://github.com/hill-a/stable-baselines/tree/master/stable_baselines).

4. Activate `virtual-env` using the `requirements.txt` file provided.

~~~
source <virtual env path>/bin/activate
~~~

# Example

Use the `run_atari.py` and `run_mujoco.py` script for training the kappa-PI/VI variants for DQN and TRPO respectively.

Kappa-PI/VI DQN
~~~
python3 run_atari.py --env=BreakoutNoFrameskip-v4
~~~

Kappa-PI/VI TRPO
~~~
python3 run_mujoco.py --env=Walker2d-v2 
~~~

# Reference

~~~
@inproceedings{tomar2020multi,
  title={Multi-step Greedy Reinforcement Learning Algorithms},
  author={Tomar, Manan and Efroni, Yonathan and Ghavamzadeh, Mohammad},
  booktitle={International Conference on Machine Learning},
  pages={9504--9513},
  year={2020},
  organization={PMLR}
}
~~~
