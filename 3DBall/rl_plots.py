import matplotlib.pyplot as plt
import os

def plot_rewards(rewards, name, save=False, first=False):
    x = [i for i in range(len(rewards))]
    
    plt.scatter(x, rewards)
    plt.xlabel('Episode number')
    plt.ylabel('Average reward past 100 games')
    
    plt.show(block=False)

    if save:
        if not os.path.isdir('./figures/'):
            os.mkdir('./figures/')

        plt.savefig('./figures/' + name + '.png')

class RLPlots(object):
    def __init__(self, name):
        self.counter = 0
        self.name = name

    def plot_rewards(self, rewards, save=False):
        x = [i for i in range(len(rewards))]

        if self.counter == 0:
            plt.ion()
            plt.show()
        
        plt.scatter(x, rewards)
        plt.xlabel('Episode number')
        plt.ylabel('Average reward past 100 games')
        plt.draw()
        plt.pause(0.001)
        
        # plt.show(block=False)

        if save:
            if not os.path.isdir('./figures/'):
                os.mkdir('./figures/')

            plt.savefig('./figures/' + self.name + '.png')


def plot_logfile(filename, pattern, xlbl=None, ylbl=None):

    log_raw = open(filename, 'r').read()
    match = re.findall(pattern, log_raw, re.M|re.I)

    # while match is not None:
    # print(match[3374:])
    y = [float(i) for i in match]
    x = [i for i in range(len(y))]

    plt.figure(figsize=(15,7))
    plt.scatter(x, y)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)