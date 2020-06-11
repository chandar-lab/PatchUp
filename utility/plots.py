"""
References of the code:
https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/plots.py
This code is responsible to plot information and save the result as a png file in the GPU Cluster.
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)


plot_from_index=-10000


def plotting(exp_dir):
    # Load the training log dictionary:
    train_dict = pickle.load(open(os.path.join(exp_dir, 'log.pkl'), 'rb'))

    ###########################################################
    ### Make the vanilla train and test loss per epoch plot ###
    ###########################################################
   
    plt.plot(np.asarray(train_dict['train_loss']), label='train_loss')
    plt.plot(np.asarray(train_dict['test_loss']), label='test_loss')

    #plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'loss.png' ))
    plt.clf()
   
    ## accuracy###
    plt.plot(np.asarray(train_dict['train_acc']), label='train_acc')
    plt.plot(np.asarray(train_dict['test_acc']), label='test_acc')
        
    #plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'acc.png' ))
    plt.clf()

   
if __name__ == '__main__':
    plotting('temop')

#PR_2