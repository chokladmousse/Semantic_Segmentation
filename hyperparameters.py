__config__ = {
    'data_provider': 'tas500v11.dataloader',
    'network': {
        'model': 'iterative',       # net, hourglass, iterative
        'base': 'resUnet',          # resnet, resUnet
        'prenet': 'x8',             # x1, x2, x4, x8
    },
    'inference': {
        'M': 4,
        'f': 32,
        'n': 1,
        'increase_ratio': 2,
        'normalization': 'layerHW', # batch, layerHW, layerCHW
        'num_class': 10,
        'inp_dim': (384, 768),
        'max_stack': 8,
    },
    'train': {
        'batchsize': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'loss': 'DiceLoss',
    },
}
