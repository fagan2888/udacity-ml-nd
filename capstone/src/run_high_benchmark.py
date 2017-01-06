#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import random

from datalogger import ExperimentLogger
from evaluator import evaluate

ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
# To get repeatable results.
sd = 16
np.random.seed(sd)
random.seed(sd)
env.seed(sd)
nb_actions = env.action_space.n

env.monitor.start('../monitor/LunarLander_HighBenchMark-1')

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


memory = SequentialMemory(limit=500000, window_length=1)
policy = EpsGreedyQPolicy(eps=1.0)

class EpsDecayCallback(Callback):
    def __init__(self, eps_poilcy, decay_rate=0.95):
        self.eps_poilcy = eps_poilcy
        self.decay_rate = decay_rate
    def on_episode_begin(self, episode, logs={}):
        self.eps_poilcy.eps *= self.decay_rate
        print 'eps = %s' % self.eps_poilcy.eps

class LivePlotCallback(Callback):
    def __init__(self, nb_episodes=500, prefix='highbenchmark'):
        self.nb_episodes = nb_episodes
        self.el = ExperimentLogger('../log', prefix, nb_episodes)
        
    def on_episode_end(self, episode, logs):
        rw = logs['episode_reward']
        steps = logs['nb_episode_steps']
        self.el.log_episode(rw, 0.0, steps)
        
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=False)
dqn.compile(Adam(lr=0.002, decay=2.25e-05), metrics=['mse'])

cbs = [EpsDecayCallback(eps_poilcy=policy, decay_rate=0.975)]
cbs += [LivePlotCallback(nb_episodes=500, prefix='highbenchmark_train')]
dqn.fit(env, nb_steps=200596, visualize=False, verbose=2, callbacks=cbs)

#dqn.save_weights('../monitor/LunarLander_HighBenchMark-1/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

env.monitor.close()

# evaluate the algorithm for 100 episodes.
cbs = [LivePlotCallback(nb_episodes=500, prefix='highbenchmark_test')]
dqn.test(env, nb_episodes=500, nb_max_episode_steps=1000, visualize=False, callbacks=cbs)
evaluate('../log/highbenchmark_test_data.csv', '../log/highbenchmark_test_evaluation.png', 'High Benchmark')

