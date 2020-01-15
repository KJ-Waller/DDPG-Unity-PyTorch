import matplotlib.pyplot as plt
import os

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

        if save:
            if not os.path.isdir('./figures/'):
                os.mkdir('./figures/')

            plt.savefig('./figures/' + self.name + '.png')
