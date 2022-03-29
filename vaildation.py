import os
import shutil
import argparse
import importlib
import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_command_line():
    """
    Parse command line and get the name of the experiment and type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, help='experiments name')
    parser.add_argument('-best','--best', type=bool, default=False, help='Use best checkpoint')
    args = parser.parse_args()
    return args


def load(config):
    opt = config['opt']

    resume = os.path.join('exp', opt.exp)
    if opt.best:
      resume_file = os.path.join(resume, 'best_checkpoint.pt')
    else:
      resume_file = os.path.join(resume, 'checkpoint.pt')
    
#     resume_file = os.path.join(resume, 'checkpoint.pt')

    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file, map_location=config['device'])
        
        config['inference'] = checkpoint['inference']
        config['network'] = checkpoint['network']

        config['inference']['net'].load_state_dict(checkpoint['state_dict'])
        config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
        config['train']['epoch'] = checkpoint['epoch']
        config['inference']['best_loss'] = checkpoint['best_loss']
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        exit(0)


def save_picture(config, output, label):
    _, pred = torch.max(output[-1], dim=0)

    plt.subplot(2,1,1)
    plt.imshow(pred.cpu())
    plt.xticks([])
    plt.yticks([])
    plt.title('Network Performance')

    plt.subplot(2,1,2)
    plt.imshow(label.squeeze().cpu())
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth')

    plt.savefig(os.path.join(config['opt'].pic_path, config['phase'] + "%02d"%(config['n']) + ".png"), bbox_inches='tight')
    plt.close()


def make_picture_minibatch(config, phase, inputs, labels):
    outputs = config['inference']['net'].model(inputs)

    for i in range(outputs.shape[0]):
      if config['n'] > config['opt'].n:
        break

      save_picture(config, outputs[i], labels[i])

      config['n'] += 1
    
    return config


def make_pictures(config, data_func, phase):
    dataloader = data_func(phase)

    config['n'] = 0
    config['phase'] = phase
    for _, (inputs, labels) in tqdm.tqdm(enumerate(dataloader), total=np.ceil(config['opt'].n/len(dataloader)).astype('int'), ascii=True):
      if config['n'] > config['opt'].n:
        break

      make_picture_minibatch(config, phase, inputs, labels)


def validation(func, data_func, config):
    return None
    
    
def init():
    opt = parse_command_line()
    exp_path = os.path.join('exp', opt.exp)
    pic_path = os.path.join(exp_path, 'pictures')
    opt.exp_path = exp_path
    opt.pic_path = pic_path

    try: os.makedirs(pic_path)
    except FileExistsError: pass

    for filename in os.listdir(pic_path):
      file_path = os.path.join(pic_path, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

    config = importlib.import_module('hyperparameters').__config__
    network = importlib.import_module('models.network')
    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=> Using", config['device'])

    func = network.make_network(config)
    load(config)

    return config


if __name__ == "__main__":
    config = init()
    print("=> Loading data")
    data_func = config['data_provider'].init(config)

    for phase in ['train','valid','test']:
      print("Start " + phase + " " + config['opt'].exp)
      make_pictures(config, data_func, phase)
    
    print("=> Completed")



