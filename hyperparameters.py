__config__ = {
    'data_provider': 'tas500v11.dataloader',
    'network': {
        'model': 'hourglass',
        'base': 'resUnet',
        'prenet': 'same',
    },
    'inference': {
        'M': 3,
        'f': 8,
        'increase_ratio': 1.5,
        'normalization': 'layer', # batch or layer
        'num_class': 10,
        'scale': 1, # scale of input in comparison to output, should be set by prenet
        'inp_dim': (384, 768),
        'oup_dim': (384, 768),
        'max_stack': 4,
    },
    'train': {
        'train_iters': 1,
        'valid_iters': 1,
        'test_iters': 1,
        'batchsize': 2,
        'input_res': (256,256),
        'output_res': (256,256),
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'loss': 'DiceLoss',
    },
}