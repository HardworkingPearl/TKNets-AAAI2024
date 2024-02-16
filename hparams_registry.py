# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from torch.utils import data
from lib import misc

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST', 'DenseDomainRotatedMNIST', 'TDGPortrait', 'TDGOcularDisease', 'TDGRotatedMNIST', 'TDGRotatedMNISTDInfo']
    CSV_DATASETS = ['TDGForestCover', 'TDGDrought', 'TDGPowerSupply']
    LINEAR_DATASETS = ['TDGRPlate', "TDGEvolCircle", "TDGEvolSine", 'TDGRPlateDInfoCat', 'TDGRPlateDInfoOnehot', 'TDGRPlateDInfoProduct', 'TDGEvolCircleDInfoCat', 'TDGEvolCircleDInfoOnehot', 'TDGEvolCircleDInfoProduct']
    
    hparams = {}
    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: True) # True
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.
    if 'TDG' in algorithm or 'ERM' in algorithm:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
        _hparam('dmatch_rate', 0., lambda r: r.choice([0.0, 1.0]))
        _hparam('multistep_rate', 0., lambda r: r.choice([0.0, 1.0]))
        # _hparam('dmatch_rate', 1., lambda r: r.choice([0.0]))
        # _hparam('multistep_rate', 1., lambda r: r.choice([0.0]))
        
    elif algorithm not in ['DANN', 'CDANN', 'DDA', 'CIDA']:
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    

    
    if 'ERM' in algorithm:
        if algorithm == "ERM_Weighted":
            _hparam('l_weight_lambda', 1.0, lambda r: r.uniform(0, 1))
    
    elif 'TDG' in algorithm:
        pass

    elif algorithm in ['DANN', 'CDANN', 'CIDA']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
        if algorithm in ['CIDA']:
            _hparam('lambda_gan', 0., lambda r: 10**r.uniform(-3, 0))
            
    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 0.001, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG" or algorithm == "MLDGProto":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))
    
    elif algorithm == "GI":
        _hparam('delta_lr', 1e-1, lambda r: 10**r.uniform(-3, 0))
        _hparam('Delta', 0.15, lambda r: r.choice([0.01, 0.05, 0.1, 0.15, 0.2])) # TODO depends on domain num
        _hparam('delta_steps', 5, lambda r: r.choice([1, 5, 10, 20]))
        _hparam('lambda_GI', 1.0, lambda r: 10**r.uniform(-2, 1))
    
    elif algorithm == "LSSAE":
        # _hparam('zv_dim', 2, lambda r: 10**r.uniform(-3, 0))
        # _hparams('source_domains', [1, 2], lambda r: 10**r.uniform(-3, 0))
        _hparam('zc_dim', 20, lambda r: r.choice([2, 5, 10, 20, 40, 80]))
        _hparam('stochastic', True, lambda r: True)
        # Params for DIVA only
        _hparam('zdy_dim', 20, lambda r: r.choice([2, 5, 10, 20, 40, 80]))
        # Params for LSSAE only
        _hparam('zw_dim', 20, lambda r: r.choice([2, 5, 10, 20, 40, 80]))
        # _hparam('zv_dim', 2, lambda r: 10**r.uniform(-3, 0))  # TODO same with class num
        _hparam('coeff_y', 0.1, lambda r: 10**r.uniform(-3, 0))
        _hparam('coeff_ts', 0.1, lambda r: 10**r.uniform(-3, 0))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif dataset in LINEAR_DATASETS:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
        _hparam('mlp_width', 2, lambda r: int(2 ** r.uniform(1, 1)))
        _hparam('mlp_depth', 1, lambda r: int(2 ** r.uniform(1, 1)))
        _hparam('pure_liner', True, lambda r: bool(r.choice([True, True])))
    elif dataset in CSV_DATASETS:
        _hparam('mlp_width', 128, lambda r: int(2 ** r.uniform(4, 8)))
        _hparam('mlp_depth', 2, lambda r: int(2 ** r.uniform(0, 3)))
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))


    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(5, 7)) )
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)) )
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(5, 7)) )


    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )


    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )


    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2) )


    return hparams

def default_hparams(algorithm, dataset):
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, seed).items()}
