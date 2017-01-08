import os
from datalogger import ExperimentLogger


class Experiment(object):

    def __init__(self, env, agent, logdir="../log", verbose=True, num_episodes=1000):
        """ Takes a gym environment and an agent as inputs
            and initialize an experiment instance """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.envname = env.__str__().split(' ')[0].lstrip('<')
        self.agentname = self.agent.name
        self.monitordir = '../monitor/' + self.envname + '_' + self.agentname
        self.prefix = self.envname + '_' + self.agentname
        self.logger_train = ExperimentLogger(logdir=logdir, prefix=(self.prefix  + "_train"), num_episodes=self.num_episodes, verbose=verbose )
        self.logger_test  = ExperimentLogger(logdir=logdir, prefix=(self.prefix + "_test"),   num_episodes=self.num_episodes, verbose=verbose )
        os.makedirs(self.monitordir)
        self.env.monitor.start(self.monitordir)

    def __del__(self):
        self.env.monitor.close()

    def run(self, testmode=False):
        """ Run num_episodes episodes on self.env with self.agent.
            It will let the agent learn only if testmode==False.
        """
        for episodeidx in xrange(self.num_episodes):
            curstate = self.env.reset()
            done = False
            loss = 0.0
            numsteps = 0
            totreward = 0.0
            while not done:
                numsteps += 1
                action = self.agent.decide(curstate, testmode=testmode)
                prevstate = curstate
                curstate, reward, done, _ = self.env.step(action)
                totreward += reward
                if not testmode:
                    self.agent.observe(prevstate, action, reward, curstate, done)
                    loss += self.agent.learn()
            if testmode:
                self.logger_test.log_episode(totreward, loss, numsteps)
            else:
                self.logger_train.log_episode(totreward, loss, numsteps)
