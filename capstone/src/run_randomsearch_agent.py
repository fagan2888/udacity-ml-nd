
from agents import RandomSearchAgent
from experiments import Experiment

import gym

seed = 16

env = gym.make('LunarLander-v2')
env.seed(seed)

ragent = RandomSearchAgent(name='RandomSearchAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=seed, stop_search_reward=210)
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=1000)

exp.run()

