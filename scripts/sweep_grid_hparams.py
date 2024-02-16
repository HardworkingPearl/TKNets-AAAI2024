import os
import shutil
import sys
sys.path.append(".")
sys.path.append("..")
from sweep_tdg import Job
import command_launchers




def gen_list_of_hp_dict(left_hparams, cur_list = []):
    """Transform {a: [], b: [], ...} into [{a: xxx, b: xxx}, ..., {}]
    """
    if len(cur_list) == 0: # first level
        keys = list(left_hparams.keys())
        first_key = keys[0]
        res_list = []
        for each_v in left_hparams[first_key]:
            res_list.append({first_key: each_v})
    else:
        keys = list(left_hparams.keys())
        first_key = keys[0]
        res_list = []
        for each_v in left_hparams[first_key]:
            for each_d in cur_list:
                each_d[first_key] = each_v
                res_list.append(each_d.copy())
    del left_hparams[first_key]
    if len(keys) == 1: return res_list
    else: return gen_list_of_hp_dict(left_hparams, cur_list=res_list)

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


if __name__ == "__main__":
    # cofing
    EXPS_PARRENT_DIR = 'EXPS'
    # EXPS_NAME = "Variations_RotatedMNIST_3D"
    # EXPS_NAME = "ERM_Drop_middle"
    # EXPS_NAME = "Drought_HPSweep"
    EXPS_NAME = "GridRotatedERMTDGShareLinearAdversal"
    LAUNCHER = 'slurm' # dummy local multi_gpu slurm 
    hparams = {
        "data_dir": ['../datasets'],
        "test_type": ['backward_test'],

        #### TDGRotatedMNIST
        "dataset": ['TDGRotatedMNIST'],
        "algorithm": ['ERM', 'TKNets'],
        "env_distance": [1, 3, 5, 7, 10, 15, 20],
        # "env_distance": [5, 7, 10],
        "env_number": [3, 5, 7, 9, 11, 13, 15, 17, 19],
        # "env_number": [9, 10, 11, 13],
        "env_sample_number": [200],
        "seed": [1, 2, 3, 4, 5],
    }
    # gen hparams list for each exps
    hparams_list = gen_list_of_hp_dict(hparams)
    
    # create jobs
    jobs = [Job(train_args, os.path.join(EXPS_PARRENT_DIR, EXPS_NAME)) for train_args in hparams_list]

    for job in jobs:
        print(job.command_str)

    for job in jobs:
        print(job)
        
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )
    
    to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
    print(f'About to launch {len(to_launch)} jobs.')
    ask_for_confirmation()
    launcher_fn = command_launchers.REGISTRY[LAUNCHER]
    Job.launch(to_launch, launcher_fn)