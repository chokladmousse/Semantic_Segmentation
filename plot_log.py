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
    parser.add_argument('-name', '--name', type=str, default="", help='Name of model')
    parser.add_argument('-av', '--average_window', type=int, default=1, help='Window for moving average of data')
    args = parser.parse_args()
    return args


def init():
    opt = parse_command_line()
    exp_path = os.path.join('exp', opt.exp)
    opt.exp_path = exp_path

    return opt

def make_loss_plot(opt, log):
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
    plt.title('Loss ' + opt.name)
    plt.grid()
    plt.ylim(0,3)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def make_performance_plot(opt, log):
    log_path = os.path.join(opt.exp_path, log)
    plot_path = os.path.join(opt.exp_path, log + ".png")
    
    # Read log file
    with open(log_path) as f:
        contents = f.read().splitlines()

    # Extract performance from each line and convert to float
    contents = np.array([[float(x) for x in line.split()[1:]] for line in contents[1:]])
    
    temp_train_loss = contents[::2]
    temp_val_loss   = contents[1::2]
    train_loss = np.empty((temp_train_loss.shape[0]-(opt.average_window-1),temp_train_loss.shape[1]))
    val_loss   = np.empty((temp_val_loss.shape[0]-(opt.average_window-1),temp_val_loss.shape[1]))

    if opt.average_window > 1:
      for i in range(train_loss.shape[1]):
        train_loss[:,i] = moving_average(temp_train_loss[:,i], n=opt.average_window)
        val_loss[:,i]   = moving_average(temp_val_loss[:,i], n=opt.average_window)

    plt.figure(figsize=(10,5))
    ax = plt.subplot(1,2,1)
    for i in range(train_loss.shape[1]):
      plt.plot(np.arange(len(train_loss)), train_loss[:,i], label = log + " " + str(i))
    ax.set_ylim(0,1)
    ax.grid()
    
    ax.legend()
    if log == 'acc':
      plt.title('Training Accuracy ' + opt.name)
    elif log == 'IoU':
      plt.title('Training mIoU ' + opt.name)
    
    ax = plt.subplot(1,2,2)
    for i in range(train_loss.shape[1]):
      plt.plot(np.arange(len(val_loss)), val_loss[:,i], label = log + " " + str(i))
    ax.set_ylim(0,1)
    ax.grid()
    
    ax.legend()
    if log == 'acc':
      plt.title('Validation Accuracy ' + opt.name)
    elif log == 'IoU':
      plt.title('Validation mIoU ' + opt.name)
    
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    opt = init()
    make_loss_plot(opt, 'log')
    make_performance_plot(opt, 'acc')
    make_performance_plot(opt, 'IoU')



