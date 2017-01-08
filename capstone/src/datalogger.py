import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

WINDOW = 100

class ExperimentLogger(object):
    """ Logs per-episode stats and produces a live reward plot """
    def __init__(self, logdir, prefix, num_episodes, verbose=True):
        self.logdir = logdir
        self.prefix = prefix
        self.verbose = verbose
        self.num_episodes = num_episodes
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self._init_plots(num_episodes)
        self._init_logger()

    def _init_logger(self):
        self.datafile = open('%s/%s_data.csv' % (self.logdir, self.prefix), 'w')
        self.datafile.write('episodeidx,reward,rewardlow,rewardavg,rewardhigh,loss,steps\n')

    def _init_plots(self, num_episodes):
        self.instrewards = np.zeros(num_episodes) - 300
        self.avgrewards  = np.zeros(num_episodes) - 300
        self.upstd       = np.zeros(num_episodes) - 300
        self.downstd     = np.zeros(num_episodes) - 300
        self.losses      = np.zeros(num_episodes)
        self.steps       = np.zeros(num_episodes)
        self.episodeidx = 0
        
        x = np.arange(1, num_episodes+1)
        plt.ion()
        self.figure = plt.figure()
        ##
        self.plt1 = plt.subplot(311)
        self.instrewards_plt = self.plt1.plot(x, self.instrewards, 'k')[0]
        self.avgrewards_plt  = self.plt1.plot(x, self.avgrewards, 'b')[0]
        self.upstd_plt       = self.plt1.plot(x, self.upstd, 'g')[0]
        self.downstd_plt     = self.plt1.plot(x, self.downstd, 'r')[0]
        self.plt1.legend([self.instrewards_plt, self.avgrewards_plt, self.upstd_plt, self.downstd_plt],
                    ['Episode reward', 'Mean reward', 'Mean + 1*stddev', 'Mean - 1*stddev'], bbox_to_anchor=(0, 1), loc='upper left', ncol=4)
        self.plt1.set_xlabel('Episodes')
        self.plt1.set_ylabel('Rewards')
        self.plt1.set_ylim(bottom=-300.0, top=350)
        self.plt1.grid(b=True, which='major', color='k', linestyle='--')
        ##
        self.plt2 = plt.subplot(312)
        self.losses_plt = self.plt2.plot(x, self.losses, 'r')[0]
        self.plt2.set_xlabel('Episodes')
        self.plt2.set_ylabel('Loss')
        self.plt2.set_ylim(bottom=0, top=350)
        self.plt2.grid(b=True, which='major', color='k', linestyle='--')
        ##
        self.plt3 = plt.subplot(313)
        self.steps_plt = self.plt3.plot(x, self.steps, 'r')[0]
        self.plt3.set_xlabel('Episodes')
        self.plt3.set_ylabel('Steps')
        self.plt3.set_ylim(bottom=0, top=1010)
        self.plt3.grid(b=True, which='major', color='b', linestyle='--')
        
    def __del__(self):
        self.datafile.close()
        self.figure.savefig('%s/%s_plots.png' % (self.logdir, self.prefix))

    def log_episode(self, reward, loss, numsteps):
        if len(self.instrewards) <= self.episodeidx:
            return
        self.instrewards[self.episodeidx] = reward
        pastidx = max(0, self.episodeidx - WINDOW)
        curmean = np.mean(self.instrewards[pastidx:self.episodeidx+1])
        curstd  = np.std(self.instrewards[pastidx:self.episodeidx+1])
        self.avgrewards[self.episodeidx]   = curmean
        self.upstd[self.episodeidx]        = curmean + curstd
        self.downstd[self.episodeidx]      = curmean - curstd
        self.losses[self.episodeidx]       = loss
        self.steps[self.episodeidx]        = numsteps
        self.instrewards_plt.set_ydata(self.instrewards)
        self.avgrewards_plt.set_ydata(self.avgrewards)
        self.upstd_plt.set_ydata(self.upstd)
        self.downstd_plt.set_ydata(self.downstd)
        self.losses_plt.set_ydata(self.losses)
        self.plt2.set_ylim(bottom=min(self.losses), top=max(self.losses))
        self.steps_plt.set_ydata(self.steps)
        plt.draw()
        plt.pause(0.01)
        if self.verbose:
            print 'Episode #%d : Reward = %.2f (%.2f, %.2f, %.2f), Loss = %f, Steps = %d' % (self.episodeidx+1,
                                                                                             reward,
                                                                                             curmean - curstd,
                                                                                             curmean,
                                                                                             curmean + curstd,
                                                                                             loss,
                                                                                             numsteps)
        row = {'episodeidx' : self.episodeidx + 1,
               'reward'     : reward,
               'rewardlow'  : curmean - curstd,
               'rewardavg'  : curmean,
               'rewardhigh' : curmean + curstd,
               'loss'       : loss,
               'steps'      : numsteps}
        self.datafile.write('%(episodeidx)d,%(reward)f,%(rewardlow)f,%(rewardavg)f,%(rewardhigh)f,%(loss)f,%(steps)d\n' % row)
        self.datafile.flush()
        self.episodeidx += 1


def testrun():
    random.seed(16)
    el = ExperimentLogger(logdir='../log', prefix='testrun', num_episodes=1000)
    reward = -200.0
    loss = 200
    steps = 500
    for _ in xrange(1000):
        el.log_episode(reward, loss, steps)
        reward += 50*(random.random() - 0.5)
        loss += random.gauss(0, 50)
        steps += random.randint(-10, 10)
        time.sleep(0.2)

if __name__ == '__main__':
    testrun()
