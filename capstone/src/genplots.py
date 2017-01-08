from evaluator import evaluate
import matplotlib.pyplot as plt
import pandas as pd

infiles = ['../log/highbenchmark_test_data.csv',
           '../monitor/LunarLander_RandomAgent-1_data.csv',
           '../log/LunarLander_BasicDQNAgent-1_test_data.csv',
           '../log/LunarLander_FullDQNAgent-exper_test_data.csv']

names = ['High Benchmark', 'Low Benchmark', 'Basic DQN', 'Improved DQN']

evalplotfiles = ['../report_src/img/highbenchmark_test_evaluation.png',
                 '../report_src/img/randomagent.png',
                 '../report_src/img/basicdqn.png',
                 '../report_src/img/fulldqn.png']

ffviz_er_plotfile = '../report_src/img/ffviz_er.png'
ffviz_en_plotfile = '../report_src/img/ffviz_en.png'

numagents = len(infiles)
numtest_episodes = 500
df_er = pd.DataFrame(columns=names, index=xrange(1,numtest_episodes+1))
df_en = pd.DataFrame(columns=names, index=xrange(1,numtest_episodes+1))


for idx in xrange(numagents):
    print '-'*70
    print 'Agent : %s' % names[idx]
    print '-'*70
    evaluate(infiles[idx], evalplotfiles[idx], names[idx])
    print '='*70
    df = pd.read_csv(infiles[idx])
    df_er[names[idx]] = df['reward'].values
    df_en[names[idx]] = df['steps'].values


axs1 = df_er.hist(bins=100, alpha=1, figsize=(10, 8), layout=(4, 1), sharex=True)
fig1 = axs1[0][0].get_figure()
#[ax[0].set_ylabel('Frequency') for ax in axs1]
fig1.suptitle('Histograms of episode rewards of each agent', fontsize=13)
fig1.text(0.5, 0.04, 'Episode reward', ha='center')
fig1.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
axs2 = df_en.hist(bins=100, alpha=1, figsize=(10, 8), layout=(4, 1), sharex=True)
fig2 = axs2[0][0].get_figure()
#[ax[0].set_ylabel('Frequency') for ax in axs2]
fig2.suptitle('Histograms of episode lengths of each agent', fontsize=13)
fig2.text(0.5, 0.04, 'Episode length', ha='center')
fig2.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
fig1.savefig(ffviz_er_plotfile)
fig2.savefig(ffviz_en_plotfile)
#plt.show()
#raw_input('press enter...')
