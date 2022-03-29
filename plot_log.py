import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def parse_command_line():
    """
    Parse command line and get the name of the experiment and type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, help='experiments name')
    parser.add_argument('-name', '--name', type=str, help='Title of plot')
    args = parser.parse_args()
    return args


def init():
    opt = parse_command_line()
    exp_path = os.path.join('exp', opt.exp)
    opt.exp_path = exp_path

    return opt


def make_plot(opt, log):
    log_path = os.path.join(opt.exp_path, log)
    plot_path = os.path.join(opt.exp_path, log + ".png")
    
    # Read log file
    with open(log_path) as f:
        contents = f.read().splitlines()

    # Extract loss from each line and convert to float
    contents = [float(line.split()[-1]) for line in contents[1:]]
    
    train_loss = contents[::2]
    val_loss = contents[1::2]

    plt.plot(np.arange(len(train_loss)), train_loss, label = "Train loss")
    plt.plot(np.arange(len(val_loss)), val_loss, label = "Validation loss")
    plt.legend()
    plt.title(opt.name)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    opt = init()
    # make_plot(opt, 'log')
    make_plot(opt, 'log_mean')




