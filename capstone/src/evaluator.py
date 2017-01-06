import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(datafile, plotfname, agentname):
    df = pd.read_csv(datafile)
    er = df["reward"].mean()
    stdr = df["reward"].std()
    en = df["steps"].mean()
    psi = 0.00647582
    sharpe = er/stdr
    print 'sharpe ratio = %.5f' % sharpe
    print 'E[N] = %.5f' % en
    #print 'psi = %.8f' % (sharpe/en)
    score = sharpe - psi*en
    print 'score = %.5f' % score
    ###
    x = np.arange(1, df.shape[0]+1)
    fig = plt.figure(figsize=(12,9))
    plt1 = plt.subplot(211)
    instrewards_plt = plt1.plot(x, df['reward'], 'k',     label='Episode reward')[0]
    avgrewards_plt  = plt1.plot(x, df['rewardavg'], 'b',  label='Mean')[0]
    upstd_plt       = plt1.plot(x, df['rewardhigh'], 'g', label='Mean + 1*st.dev' )[0]
    downstd_plt     = plt1.plot(x, df['rewardlow'], 'r',  label='Mean - 1*st.dev')[0]
    plt1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=4)
    plt1.set_xlabel('Episodes')
    plt1.set_ylabel('Rewards')
    plt1.grid(b=True, which='major', color='k', linestyle='--')
    plt2 = plt.subplot(212)
    steps_plt = plt2.plot(x, df['steps'], 'r')[0]
    plt2.set_xlabel('Episodes')
    plt2.set_ylabel('Episode length')
    plt2.grid(b=True, which='major', color='k', linestyle='--')
    fig.tight_layout()
    plt.suptitle('%s agent score = %.4f' % (agentname, score), fontsize=15)
    plt.subplots_adjust(top=0.92)
    #print "psi = %.5f" % (1.0/score)
    fig.savefig(plotfname)
