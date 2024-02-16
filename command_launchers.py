# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
import uuid



def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def slurm_launcher(commands):
    """
    start a slurm job and run commands.
    """
    for command in commands:
        slurm_run_one_cmd(command)


def slurm_run_one_cmd(command):
    def readfile(path):
        with open(path, "r") as f:
            return f.read()

    def writefile(path, ctt):
        with open(path, "w") as f:
            f.write(ctt)
    
    def deletefile(path):
        os.remove(path)

    # init
    base_dir = "./slurm_scripts"
    slot_dict = {
        "###bashfile_cmd_slot###": command,
    }
    print("try to run: "+command+"...")
    # read template    
    bashfile_template = readfile(os.path.join(base_dir, "slurm_launcher_tp.sh")) # TODO
    # replace
    for slot, content in slot_dict.items():
        cur_ctt = bashfile_template.replace(slot, content)
    # create .sh and submit job
    tmp_file = str(uuid.uuid4())+'.sh'
    writefile(os.path.join(base_dir, tmp_file), cur_ctt)
    subprocess.call("sbatch "+os.path.join(base_dir, tmp_file), shell=True)
    # delete .sh
    deletefile(os.path.join(base_dir, tmp_file))
    print("Finish!")
    

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()



REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm': slurm_launcher
}

try:
    import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
