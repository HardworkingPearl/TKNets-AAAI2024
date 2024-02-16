# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
sys.path.append(".")
import time
import uuid


import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import pdb

import datasets
import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    '############# 1. load parameters'
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--test_type', type=str, default='backward_test') # for TDG 
    parser.add_argument('--env_distance', type=int, default=10) # for dataset split
    parser.add_argument('--env_number', type=int, default=12) # for dataset split
    parser.add_argument('--env_sample_number', type=int, default=200) # for dataset split
    parser.add_argument('--total_sample_number', type=int, default=0) # for dataset split
    parser.add_argument('--env_sample_ratio', type=float, default=0.5) # for dataset split
    args = parser.parse_args()
    print(args)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    '############# 2. prepare std saver'
    '''
    the misc.Tee is a wrapper, which include normal std and an extra file writer
    so if we use it to replace sys.std, then we can get a upgrated std which can
    write to the out.txt when we use 'print'
    '''
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt')) # default is /train_output/out.txt
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt')) # default is /train_output/err.txt

    '############# 3. print Environment, Args, HParams'
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    hparams['test_type'] = args.test_type
    hparams['env_distance'] = args.env_distance
    hparams['env_number'] = args.env_number
    hparams['env_sample_number'] = args.env_sample_number
    hparams['env_sample_ratio'] = args.env_sample_ratio
    hparams['total_sample_number'] = args.total_sample_number # higher priority than env_sample_number, 0 is None. env_sample_number =  total_sample_number / env_number

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    '############# 4. init numpy, torch, seed, cuda, '
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True # here is to make cuda deterministic
    torch.backends.cudnn.benchmark = False # this is an trick of speed up when setted to true. see more: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    '############# 5. get dataset'
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams) # get class by name
    else:
        raise NotImplementedError
    print(f"Dataset(all) - Size of {len(dataset.ENVIRONMENTS)} datasets: ")
    misc.print_row(dataset.ENVIRONMENTS,
        colwidth=12)
    misc.print_row([len(each) for each in dataset], colwidth=12)
    

    '############# 6. setup TDG'
    if args.test_type == 'forward_test': 
        args.test_envs = [len(dataset) - 1]
        args.query_env = len(dataset) - 1
        args.support_env = len(dataset) - 2
    elif args.test_type == 'backward_test':
        args.test_envs = [0]
        args.query_env = 0
        args.support_env = 1
    elif args.test_type == 'forward_val':
        args.test_envs = [len(dataset) - 1, len(dataset) - 2]
        args.query_env = len(dataset) - 1
        args.support_env = len(dataset) - 2
    elif args.test_type == 'backward_val':
        args.test_envs = [0, 1]
        args.query_env = 1
        args.support_env = 2
    elif args.test_type == 'middle':
        args.test_envs = [int(len(dataset)/2)]
        args.query_env = int(len(dataset)/2)
        args.support_env = int(len(dataset)/2-1)
    else:
        raise NotImplementedError(f"not implemented test_type: {args.test_type}")

    '############# 7. split data in each env + get weights' # split item in env list into in_splits=[(in_, in_weights), ...], out_splits=[(out_, out_weights), ...], uda_splits=[(uda_, uda_weights), ...]
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # print(env.)
        # split # if train_env: env->(out+in_); if test_env: env->(out+in_+uda) ps. in_ = uda + in_
        out, in_ = misc.split_dataset(env, # ? what is out and in? can be treat as train and validation
            int(len(env)*args.holdout_fraction), # TODO what is holdout_fraction
            misc.seed_hash(args.trial_seed, env_i))
        if(env_i == args.support_env):
            support_dataset = in_

        if(env_i == args.query_env):
            query_dataset = in_
        
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                0,
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']: # make weight # make weight for out, in_, uda, no balanced sampling when setted to None
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
    print(f"Dataset(in_splits) - Size of {len(in_splits)} datasets: ")
    misc.print_row(dataset.ENVIRONMENTS,
        colwidth=12)
    misc.print_row([len(in_) for (in_, in_weights) in in_splits], colwidth=12)
    print(f"Dataset(out_splits) - Size of {len(out_splits)} datasets: ")
    misc.print_row(dataset.ENVIRONMENTS,
        colwidth=12)
    misc.print_row([len(out) for (out, out_weights) in out_splits], colwidth=12)

    '############# 8. make loader for train + support + query' # train=in of non-test, uda=uda of test, eval=in_+out+uda of all
    train_loaders = [InfiniteDataLoader( # why ininite?
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs] # 
    print(f"Dataset(train_loaders) - Size of {len(in_splits)-len(args.test_envs)} datasets: ")
    misc.print_row([dataset.ENVIRONMENTS[i] for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs],
        colwidth=12)
    misc.print_row([len(env) for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs], colwidth=12)

    eval_support_loaders = [FastDataLoader(
        dataset=support_dataset,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS
    )] # add s for multiple expanding
    print(f"Dataset(eval_support_loaders) - Size dataset: ")
    misc.print_row([dataset.ENVIRONMENTS[args.support_env]],
        colwidth=12)
    misc.print_row([len(support_dataset)], colwidth=12)

    eval_query_loader = FastDataLoader(
        dataset=query_dataset,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS
    )
    print(f"Dataset(eval_query_loader) - Size of dataset: ")
    misc.print_row([dataset.ENVIRONMENTS[args.query_env]],
        colwidth=12)
    misc.print_row([len(query_dataset)], colwidth=12)

    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    '############# 9. init algorithm, train/uda iterator, checkpoint'
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, # input_shape, num_classes, num_domains, hparams
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None: # load state
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders) # note: zip(*iterables), the result iterator will return a tuple as a whole parallelly. then we can use 
    checkpoint_vals = collections.defaultdict(lambda: []) # store loss, acc, step, ... from update # after write to result, this is set to empty again

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        algorithm.to(device)
        torch.save(save_dict, os.path.join(args.output_dir, filename))
        print("save model to: "+os.path.join(args.output_dir, filename))


    last_results_keys = None
    '############# 10. main loop'
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)] # sample from train loader and to device # [sample from each env, ...]
        ### update
        step_vals = algorithm.update(minibatches_device, None) # samples -> algorithm.update()
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items(): # save log. split val and save them in a dict of list, each key is a metric
            checkpoint_vals[key].append(val)

        '############# 11. checkpoint eval+log'
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items(): # calculate mean from checkpoint buffer then store them in results
                results[key] = np.mean(val)
                # print("{}: {}".format(key, results[key]))

            # evals = zip(eval_loader_names, eval_loaders, eval_weights)
            # for name, loader, weights in evals:
            #     acc = misc.accuracy(algorithm, loader, weights, device)
            #     results[name+'_acc'] = acc
            '############# 12. eval'

            # if 'TDG' in args.algorithm:
            acc = misc.tdg_accuracy(algorithm, eval_support_loaders, eval_query_loader, device, args)
            results['query_acc'] = acc
            # else:
            #     eval_report = misc.accuracy(algorithm, eval_support_loaders, None, device)   # eval_weights
            #     results.update(eval_report)
            # elif args.algorithm in ["DDA", "CIDA"]:
            #     support_acc = misc.dda_accuracy(algorithm, eval_support_loaders[0], None, device, 0)
            #     query_acc = misc.dda_accuracy(algorithm, eval_query_loader, None, device, -1)
            #     results['support_acc'] = support_acc
            #     results['query_acc'] = query_acc
            # else:
            #     support_acc = misc.accuracy(algorithm, eval_support_loaders[0], None, device)
            #     query_acc = misc.accuracy(algorithm, eval_query_loader, None, device)
            #     results['support_acc'] = support_acc
            #     results['query_acc'] = query_acc
            # else:
            #     raise NotImplementedError("not implemented algorithm")

            '############# 13. print results in two rows'
            results_keys = sorted(results.keys())
            # if results_keys != last_results_keys:
            print("===== eval results")
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


