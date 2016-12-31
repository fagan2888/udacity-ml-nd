
from agents import RandomAgent
from experiments import Experiment

import gym

seed = 16

env = gym.make('LunarLander-v2')
env.seed(seed)

ragent = RandomAgent(name='RandomAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=seed)
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=500)

exp.run()

mean, std = ragent.describe_state_variables()

print 'Mean of observed states ='
print mean

print 'Standard deviation of observed states ='
print std
