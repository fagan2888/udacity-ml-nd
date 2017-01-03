
from agents import RandomAgent
from experiments import Experiment
import matplotlib.pyplot as plt
import pickle

import gym

seed = 16

env = gym.make('LunarLander-v2')
env.seed(seed)

ragent = RandomAgent(name='RandomAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=seed)
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=500)

exp.run()

mean, std = ragent.describe_state_variables()

fig = plt.figure()
plt.subplot(211)
plt.bar(list(xrange(0,8)), mean)
plt.xlabel('State space components')
plt.ylabel('Mean')
plt.title('Mean of state space components')

plt.subplot(212)
plt.bar(list(xrange(0,8)), std)
plt.xlabel('State space components')
plt.ylabel('Standard deviation')
plt.title('Standard deviation of state space components')

fig.tight_layout()

print 'Mean of observed states ='
print mean

print 'Standard deviation of observed states ='
print std

pickle.dump( (mean, std), open( "../params/state-stats.pkl", "wb" ) )

fig.savefig('../log/state-space.png')

plt.show()
