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
        self.monitordir = self.envname + '_' + self.agentname
        self.logger = ExperimentLogger(logdir=logdir, verbose=verbose, prefix=self.monitordir)
        os.makedirs(self.monitordir)
        self.env.monitor.start(self.monitordir)

    def __del__(self):
        self.env.monitor.close()

    def run(self, learn=True):
        """ Run num_episodes episodes on self.env with self.agent.
            It will let the agent learn only if learn==True.
        """
        for episodeidx in xrange(self.num_episodes):
            curstate = self.env.reset()
            done = False
            loss = 0.0
            numsteps = 0
            totreward = 0.0
            while not done:
                numsteps += 1
                action = self.agent.decide(curstate)
                prevstate = curstate
                curstate, reward, done, _ = self.env.step(action)
                self.agent.observe(prevstate, action, reward, curstate, done)
                totreward += reward
                if learn:
                    loss += self.agent.learn()
            self.logger.log_episode(totreward, loss, numsteps)
                
            
