# Generalizing across Temporal Domains with Koopman Operators

[__[Paper]__](https://arxiv.org/abs/2402.07834) 
&nbsp; 
This is the authors' official PyTorch implementation for Temporal Koopman Networks (TKNets) method in the **AAAI 2024** paper [Generalizing across Temporal Domains with Koopman Operators](https://arxiv.org/abs/2402.07834.pdf).


## Prerequisites
- PyTorch >= 1.12.1 (with suitable CUDA and CuDNN version)
- torchvision >= 0.10.0
- Python3
- Numpy
- pandas

## Dataset
Dataset can be downloaded using "datasets.py"


## Training and test
Experiments on all datasets
```
python -m scripts.sweep_tdg launch\
       --data_dir=../[dataset folder]\
       --output_dir=./EXPS/AllDataset\
       --command_launcher slurm\
       --algorithms TKNets\
       --datasets TDGRPlate TDGEvolCircle TDGRotatedMNIST TDGPortrait TDGForestCover\
       --n_hparams 10\
       --n_trials 5

```

Run an experiment on Portrait dataset
```
python scripts/train_tdg.py --data_dir /datasets --algorithm TKNets --dataset TDGPortrait --test_type forward_test --seed 2 --output_dir /EXPS --save_model_every_checkpoint

```



## Acknowledgement
This code is implemented based on the [domainbed](https://github.com/facebookresearch/DomainBed) code.

## Citation
If you use this code for your research, please consider citing:
```
@article{zeng2024generalizing,
  title={Generalizing across Temporal Domains with Koopman Operators},
  author={Zeng, Qiuhao and Wang, Wei and Zhou, Fan and Xu, Gezheng and Pu, Ruizhi and Shui, Changjian and Gagne, Christian and Yang, Shichun and Wang, Boyu and Ling, Charles X},
  journal={arXiv preprint arXiv:2402.07834},
  year={2024}
}
```