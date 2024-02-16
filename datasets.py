# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing_extensions import dataclass_transform
from numpy.lib.function_base import angle
from pathlib import Path
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import scipy.io
import pandas as pd
import pandas
import datetime
from tqdm import tqdm
from scipy.io import loadmat

import numpy as np
import numpy
import shutil
import pickle
import pdb
import pandas
import numpy
from datetime import datetime

import argparse
import tarfile
import gdown
import uuid
import json
import os
from zipfile import ZipFile
# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # TDG
    "TDGRPlate",
    "TDGEvolCircle",
    "TDGRotatedMNIST",
    "TDGPortrait",
    "TDGForestCover",
    "TDGDrought",
    "TDGRPlateDInfoCat",
    "TDGRPlateDInfoOnehot",
    "TDGRPlateDInfoProduct",
    "TDGEvolCircleDInfoCat",
    "TDGEvolCircleDInfoOnehot",
    "TDGEvolCircleDInfoProduct",
    "TDGRotatedMNISTDInfo",
    "TDGPowerSupply",
    "TDGCalTran"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 0            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        "get data from MINIST and concat tr, te, then shuffle, split into 8 domains"
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            if not hasattr(self, 'env_sample_number'):
                'origin'
                images = original_images[i::len(environments)]
                labels = original_labels[i::len(environments)]
            else:
                'specific sample number'
                images = original_images[i *
                                        self.env_sample_number:(i+1)*self.env_sample_number]
                labels = original_labels[i *
                                        self.env_sample_number:(i+1)*self.env_sample_number]
            'all data, replicate'
            # images = original_images
            # labels = original_labels
            'all data / 10, replicate'
            # images = original_images[10::10]
            # labels = original_labels[10::10]
            # images = images[10::10]
            # labels = labels[10::10]
            self.datasets.append(dataset_transform(
                images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, tuple(map(lambda x: int(x), self.ENVIRONMENTS)),
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)



class TDGCombineDataset():
    '''
    random sample one from last d then cat with then return
    '''

    def __init__(self, d, last_d):
        self.d = d
        self.last_d = d

    def __getitem__(self, i):
        '''random sample one from the last then stack and return'''
        rand_i = 0

        # res = torch.stack((, self.d[i]), dim=0)
        return {
            'support': self.last_d[rand_i],
            'query': self.d[i],
        }

    def __len__(self):
        return len(self.d)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

def array_outer_product(A, B, result=None):
    ''' Compute the outer-product in the final two dimensions of the given arrays.
    If the result array is provided, the results are written into it.
    '''
    assert(A.shape[:-1] == B.shape[:-1])
    if result is None:
        result=scipy.zeros(A.shape+B.shape[-1:], dtype=A.dtype)
    if A.ndim==1:
        result[:,:]=scipy.outer(A, B)
    else:
        for idx in range(A.shape[0]):
            array_outer_product(A[idx,...], B[idx,...], result[idx,...])
    return result

class SimpleSyntheticDataset(MultipleDomainDataset):
    """Read synthetic dataset.

    Read synthetic dataset of the from data_dir, it should be a pickle
    path such as Path(data_dir) / 'RPlate/data/RPlate.pkl'.
    The data should be dict like this:
        {
        'data': [], # each element is for one domain 
            # (SAMPLE_NUM_EACH_DOMAIN, 2)
        'label': [], # each element is for one domain 
            # (SAMPLE_NUM_EACH_DOMAIN)
        'domain': [] # [[0, 0, ..., 0], [1, 1, ..., 1], ..., [M, M, ..., M]] 
            # (SAMPLE_NUM_EACH_DOMAIN)
        }
    domain_num, self.input_shape, self.num_classes, self.ENVIRONMENTS are
    automatically generated based on the dataset content.
    """

    def __init__(self, data_dir, test_envs, hparams, domain_info=None):
        self.data_dir = data_dir
        # load data
        data_pkl = self.load_data(data_dir)
        # config
        domain_num = len(list(set(data_pkl['domain'])))
        self.num_classes = len(list(set(data_pkl['label'])))
        self.ENVIRONMENTS = ['Domain '+ str(i) for i in range(domain_num)]
        # convert to torch Dataset
        self.datasets = []
        for d in range(domain_num):
            # get x, y from data_pkl
            idx = data_pkl['domain'] == d
            x = data_pkl['data'][idx].astype(np.float32)
            if domain_info is None:
                self.input_shape = data_pkl['data'][0].shape
            elif domain_info == 'cat':
                tmp = np.empty((x.shape[0], 1))
                tmp.fill(float(d)/domain_num)
                x = np.concatenate((x, tmp), axis=1)
                self.input_shape = (data_pkl['data'][0].shape[0] + 1, )
            elif domain_info == 'onehot':
                tmp = np.empty((x.shape[0], )) # (1000, )
                tmp.fill(d) # (1000, )
                tmp = tmp.astype(int)
                tmp = np.eye(domain_num)[tmp]
                # import pdb; pdb.set_trace()
                # tmp_[np.arange(x.shape[0]), tmp] = 1
                x = np.concatenate((x, tmp), axis=1)
                self.input_shape = (data_pkl['data'][0].shape[0] + domain_num, )
            elif domain_info == 'product':
                tmp = np.empty((x.shape[0], )) # (1000, )
                tmp.fill(d) # (1000, )
                tmp = tmp.astype(int)
                tmp = np.eye(domain_num)[tmp]
                # import pdb; pdb.set_trace()
                x = array_outer_product(x, tmp)
                x = x.reshape(x.shape[0], -1)
                # (1000, a) (1000, b) (1000, a, b)
                self.input_shape = (data_pkl['data'][0].shape[0] * domain_num, )
            else:
                raise NotImplementedError
            y = data_pkl['label'][idx].astype(np.int64)
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            self.datasets.append(TensorDataset(torch.tensor(x).float(), y))

    def load_data(self, path=None):
        if not path: raise NotImplementedError
        return self.read_pickle(path)

    def read_pickle(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data


class TDGRPlate(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGRPlate, self).__init__(
            Path(data_dir) / 'RPlate/data/RPlate.pkl', 
            test_envs, hparams)

class TDGRPlateDInfo(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGRPlateDInfo, self).__init__(
            Path(data_dir) / 'RPlate/data/RPlate.pkl', 
            test_envs, hparams, domain_info='product')
        
class TDGRPlateDInfoCat(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGRPlateDInfoCat, self).__init__(
            Path(data_dir) / 'RPlate/data/RPlate.pkl', 
            test_envs, hparams, domain_info='cat')
    
class TDGRPlateDInfoOnehot(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGRPlateDInfoOnehot, self).__init__(
            Path(data_dir) / 'RPlate/data/RPlate.pkl', 
            test_envs, hparams, domain_info='onehot')
    
class TDGRPlateDInfoProduct(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGRPlateDInfo, self).__init__(
            Path(data_dir) / "RPlate/data/RPlate.pkl", 
            test_envs, hparams, domain_info='product')
    
class TDGEvolCircle(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGEvolCircle, self).__init__(
            Path(data_dir) / "toy-circle/data/half-circle.pkl", 
            test_envs, hparams)

class TDGEvolCircleDInfo(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGEvolCircleDInfo, self).__init__(
            Path(data_dir) / 'toy-circle/data/half-circle.pkl', 
            test_envs, hparams, domain_info='product')

class TDGEvolCircleDInfoCat(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGEvolCircleDInfoCat, self).__init__(
            Path(data_dir) / 'toy-circle/data/half-circle.pkl', 
            test_envs, hparams, domain_info='cat')

class TDGEvolCircleDInfoOnehot(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGEvolCircleDInfoOnehot, self).__init__(
            Path(data_dir) / 'toy-circle/data/half-circle.pkl', 
            test_envs, hparams, domain_info='onehot')
        
class TDGEvolCircleDInfoProduct(SimpleSyntheticDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super(TDGEvolCircleDInfoProduct, self).__init__(
            Path(data_dir) / 'toy-circle/data/half-circle.pkl', 
            test_envs, hparams, domain_info='product')


class TDGRotatedMNIST(RotatedMNIST):
    ''' Spawn envs based on env_distance, env_number, env_sample number.
    '''
    ENVIRONMENTS = []

    def __init__(self, root, test_envs, hparams={'env_distance': 10, 'env_number': 12, 'env_sample_number': 200}):
        # convert hparams[env_distance, env_number, env_sample_number]
        # convert env_distance, env_number -> self.ENVIRONMENTS (pass as parameter)
        # convert env_sample_number -> self.env_sample_number (pass as attribute)

        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(hparams['env_number'])]
        self.env_sample_number = hparams['env_sample_number']

        # invoke RotatedMNIST
        super(RotatedMNIST, self).__init__(root, tuple(map(lambda x: int(x), self.ENVIRONMENTS)),
                                           self.rotate_dataset, (1, 28, 28,), 10)

class DomainDataset(Dataset):
    def __init__(self, dataset, d_idx):
        self.dataset = dataset
        self.d_idx = d_idx

    def __len__(self, ):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        res_ = torch.full_like(x, self.d_idx)
        # import pdb; pdb.set_trace()
        return torch.cat((x, res_), dim=0), y

class TDGRotatedMNISTDInfo(TDGRotatedMNIST):
    def __init__(self, root, test_envs, hparams={'env_distance': 10, 'env_number': 12, 'env_sample_number': 200}):
        super(TDGRotatedMNISTDInfo, self).__init__(
            root, test_envs, hparams=hparams)
        self.input_shape = (self.input_shape[0]*2, *self.input_shape[1:])
        self.datasets = [DomainDataset(dataset, i) for i, dataset in enumerate(self.datasets)]
            
class TDGPortrait(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        # load data
        original_images, original_labels = self.load_portraits_data(data_dir)
        # calculate attrs
        total_sample_number = len(original_images)
        self.env_number = hparams['env_number']
        self.env_interval = int(total_sample_number / self.env_number)
        self.env_sample_number = int(self.env_interval * hparams['env_sample_ratio'])
        self.ENVIRONMENTS = ["Domain_{}".format(self.env_interval * i)
                             for i in range(self.env_number)]
        self.input_shape = (1, 32, 32, )
        self.num_classes = 2
        # split and append to self.datasets
        self.datasets = []
        for i in range(len(self.ENVIRONMENTS)):
            images = original_images[i*self.env_interval:i*self.env_interval+self.env_sample_number]
            labels = original_labels[i*self.env_interval:i*self.env_interval+self.env_sample_number]
            x_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()])
            x = torch.zeros(len(images), 1, 32, 32)
            for i in range(len(images)):
                x[i] = x_trans(images[i])
            y = torch.tensor(labels).view(-1).long()
            self.datasets.append(TensorDataset(x, y))

    def load_portraits_data(self, data_dir):
        """Total sample number 37921, order by year.
        """
        data = scipy.io.loadmat(os.path.join(data_dir, 'portrait_dataset_32x32.mat'))
        return data['Xs'], data['Ys'][0]


class TDGCalTran(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        num_domains = 34
        self.Environments = np.arange(num_domains)
        self.ENVIRONMENTS = self.Environments
        self.input_shape = (3, 32, 32, )
        self.num_classes = 10
        root = Path(root) / "CalTrain"
        dataset_transform = None
        if dataset_transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[-2:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = dataset_transform

        if root is None:
            raise ValueError('Data directory not specified!')
        dict_path = os.path.join(root, 'caltran_dataset_labels.mat')
        data_dict = loadmat(dict_path)

        img_names = data_dict['names']
        img_labels = data_dict['labels'][0]
        img_labels = [0 if item <= 0 else 1 for item in img_labels]
        img_names = [np.array2string(item)[9:25]+'.jpg' for item in img_names]

        self.datasets = []

        pre_idx = 0

        for i in range(2, len(self.Environments) + 2):
            data_idx = i // 3
            part_idx = i % 3

            cur_day_image_list = list(filter(lambda k: k.startswith('2013-03-{:02}'.format(4+data_idx)), img_names))
            if part_idx == 0:
                cur_image_list = list(filter(lambda k: k[11:-4] <= '08-00', cur_day_image_list))
            elif part_idx == 1:
                cur_image_list = list(filter(lambda k: '08-00' < k[11:-4] < '16-00', cur_day_image_list))
            else:
                cur_image_list = list(filter(lambda k: k[11:-4] >= '16-00', cur_day_image_list))
            cur_labels = img_labels[pre_idx: pre_idx+len(cur_image_list)]
            # print('Domain idx:{}, image num:{}, label num:{}'.format(i-2, len(cur_image_list), len(cur_labels)))
            pre_idx += len(cur_image_list)
            self.datasets.append(SubCalTran(root, cur_image_list, cur_labels, self.transform))

class MultipleEnvironmentPowerSupply(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, test_envs, hparams):
        super().__init__()
        num_domains = 30
        # self.Environments = environments
        self.Environments = np.arange(num_domains)
        self.root = Path(root) / "powersupply.arff"
        self.input_shape = (2, ) # TODO
        self.num_classes = 2 # TODO
        self.normalize = True

        self.drift_times = [17, 47, 76]
        self.num_data = 29928  # original data number

        self.X, self.Y = self.load_data()
        # normalize
        if self.normalize:
            self.X = self.X / np.max(self.X, axis=0)

        self.datasets = []
        for i in range(len(environments)):
            images = self.X[i::len(environments)]
            labels = self.Y[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels))


    def load_data(self):
        X = []
        Y = []
        with open(self.root) as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if int(fields[2]) < 12 else 0

                cur_x = np.array([float(fields[0]), float(fields[1])], np.float32)

                X.append(cur_x)
                Y.append(label)
                i += 1
        assert len(X) == self.num_data
        return np.array(X), np.array(Y, np.int64)

class TDGPowerSupply(MultipleEnvironmentPowerSupply):
    def __init__(self, root, test_envs, hparams):
        num_domains = 30
        environments = list(np.arange(num_domains))
        self.ENVIRONMENTS = environments

        super().__init__(root, environments, self.process_dataset, test_envs, hparams)

    def process_dataset(self, data, labels):
        x = torch.tensor(data).float()
        y = torch.tensor(labels).long()
        return TensorDataset(x, y)



def RGB_loader(path):
    return Image.open(path).convert('RGB')

class SubCalTran(Dataset):
    def __init__(self, root, image_name_list, image_labels, dataset_transform, loader=RGB_loader):
        super(SubCalTran, self).__init__()
        self.root = root
        self.image_name_list = image_name_list
        self.image_labels = image_labels
        self.transform = dataset_transform
        self.loader = loader

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_name_list[idx])
        label = self.image_labels[idx]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_name_list)

class CsvDataset(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        return
    
    def split_domains(self, df, col, num, ratio, min_=None, max_=None):
        if min_ is None: min_ = df[col].min()
        if max_ is None: max_ = df[col].max()
        df = df.sort_values(by = col, ascending = True)
        bins = np.arange(min_, max_, (max_ - min_)/num)
        se1 = pd.cut(df[col], bins)
        df = df.drop(col, axis=1)
        gb = df.groupby(se1)
        gbs = [gb.get_group(x) for x in gb.groups]
        for each in gbs:
            print(each.groupby('label').size())
        gbs = [self.get_xy_from_df(each[:int(each.shape[0]*ratio)]) for each in gbs]
        # to datasets
        datasets = []
        for x, y in gbs:
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            datasets.append(TensorDataset(torch.tensor(x).float(), y))
        return datasets
    
    def load_data(self, path = None):
        df = pd.read_csv(os.path.join(self.data_dir, path))
        return df

    def pre_process(self, df):
        return df
    
    def get_xy_from_df(self, df):
        Y = df['label'].to_numpy()
        X = df.drop('label', axis='columns').to_numpy()
        return (X, Y)

class TDGForestCover(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        self.data_dir = data_dir
        if 'dm_idx' in hparams.keys():
            self.dm_idx = hparams['dm_idx']
        else:
            self.dm_idx = False

        COL = 'Elevation'
        MAX = 3451  # df[COL].max()
        MIN = 2061  # df[COL].min()
        COUNT = hparams['env_number'] + 1

        # pre
        self.datasets = []
        # df = self.load_forestcover_data().drop('Id', axis = 1)
        df = self.load_forestcover_data()
        # MAX = df[COL].max() # 3451 # df[col].max()
        # MIN = df[COL].min() # 2061 # df[col].min()
        bins = np.arange(MIN, MAX, (MAX - MIN) / COUNT)
        se1 = pd.cut(df[COL], bins)
        df = df.drop(COL, axis=1)
        gb = df.groupby(se1)
        gbs = [gb.get_group(x) for x in gb.groups]
        # groupby('Cover_Type').size()
        for each in gbs:
            print(each.groupby('label').size())
        gbs = [self.get_xy_from_df(each) for each in gbs]
        for i, (x, y) in enumerate(gbs):
            y = torch.tensor(y).view(-1).long()  # turn  1, 2, 3 to 0, 1, 2
            # print(y)
            # normalize the input x
            x = torch.nn.functional.normalize(torch.tensor(x).float(), dim=0)
            if self.dm_idx:
                self.datasets.append(TensorDataset(x, y, torch.tensor([float(i)] * len(x))))
            else:
                self.datasets.append(TensorDataset(x, y))
        self.input_shape = (54,)
        self.num_classes = 2
        self.ENVIRONMENTS = [str(hparams['env_distance'] * i)
                             for i in range(COUNT - 1)]
        return

    def load_forestcover_data(self, path='ForestCover/train.csv'):
        df = pd.read_csv(os.path.join(self.data_dir, path))
        df = df.rename(columns={"Cover_Type": "label"})
        df = self.group_labels(df)
        df = df.drop('Id', axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=False)  # [index, label]
        return df

    def group_labels(self, df):
        groups = [
            [0, 1, 6, 3],
            [4, 5, 2, 7],
        ]

        # print(df)

        def new_label(row):
            for new_l in range(len(groups)):
                if row['label'] in groups[new_l]:
                    return new_l

        df['label'] = df.apply(new_label, axis=1)
        # print(df)
        return df

    def get_xy_from_df(self, df):
        Y = df['label'].to_numpy()
        X = df.drop('label', axis='columns').to_numpy()
        return (X, Y)

def TDGDrought_add_label():
    DATA_DIR = '../datasets'
    FILE_PATH = 'drought/train_timeseries.csv'
    OUTPUT_PATH = 'drought/train_timeseries_with_label.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_PATH))
    # fill all with labels, to int, remove score
    label = df['score'].to_numpy()
    df = df.drop('score', axis='columns')
    last = 0
    for i in tqdm(reversed(range(label.shape[0]))):
        if np.isnan(label[i]): label[i] = last
        else: last, label[i] = int(label[i]), int(label[i])
    df['label'] = label
    df.to_csv(os.path.join(DATA_DIR, OUTPUT_PATH))
    print(f'save train_timeseries_with_label to {os.path.join(DATA_DIR, OUTPUT_PATH)}')

def TDGDrought_split():
    DATA_DIR = '../datasets'
    FILE_PATH = 'drought/train_timeseries_with_label.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_PATH))
    # fill all with labels, to int, remove score
    for num in [5, 50, 500]:
        tar_path = os.path.join(DATA_DIR, f'drought/train_{num}k.csv')
        df_tmp = df.sample(n = 1000*num)
        df_tmp.to_csv(tar_path)
        print(f'save data to {tar_path}')

class TDGDrought(CsvDataset):

    ENVIRONMENTS = []

    def __init__(self, data_dir, test_envs, hparams):
        super().__init__(data_dir, test_envs, hparams)
        # conifg
        hparams['env_number'] = 40
        hparams['env_sample_ratio'] = 0.8

        # config
        FILE_PATH = 'drought/train_500k.csv'
        TO_PROCESS = False if 'processed' in FILE_PATH else True

        # load date
        df = self.load_data(FILE_PATH)
        
        print(f'load csv of {df.shape[0]} samples.')
        # print(df)

        if TO_PROCESS: df = self.pre_process(df)
        
        self.datasets = self.split_domains(df, 'date', hparams['env_number'], hparams['env_sample_ratio'], min_=None, max_=None)

        # config
        self.input_shape = (20, )
        self.num_classes = 6
        self.ENVIRONMENTS = [str(i) for i in range(hparams['env_number'])]
        print('TDGDrought init finished.')

    def pre_process(self, df):
        """ [features_1, features_2, ..., label]
        """
        def mean_norm(df_input):
	        return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        # date to days
        df['date'] = df['date'].apply(pd.to_datetime)
        df['date'] = df['date'] - df['date'].min()
        df['date'] = df['date'].dt.days
        # print('to_days')
        # remove data
        df = df.drop('Unnamed: 0', axis='columns')
        # norm
        # print('start to mean norm ...')
        label = df['label']
        df = mean_norm(df)
        df['label'] = label
        return df


    # get the domain from year by year
    # return { year :
    #                   { region :
    #                               [ start row index , end_row index (not included), length]} }
    #
    def orgnize_domain_dict(self, dataframe):
        current_year = str(dataframe['date'].iloc[0])[0:4]
        current_area = dataframe['fips'].iloc[0]
        dataframe_len = len(dataframe)
        last_domain_index = 0
        for i in range(1,dataframe_len):
            this_year = str(dataframe['date'].iloc[i])[0:4]
            #print(" this year", this_year, " index ", i)
            this_area = dataframe['fips'].iloc[i]
            if this_year != current_year:
                # from last_domain_index to the i-th, i-th not included
                self.domain_dict[current_year] = {}
                self.domain_dict[current_year][current_area] = [last_domain_index, i, i-last_domain_index]
                last_domain_index = i
                current_year = this_year
                current_area = this_area
                #print(" year  ", current_year)
                #print(" area ", current_area)
                #print(" dict length", len(self.domain_dict))
            if i == (dataframe_len-1):
                self.domain_dict[current_year][current_area] = [last_domain_index, i+1, i+1-last_domain_index]

        return self.domain_dict

    #
    # set the domain as the area
    #
    def orgnize_domain_by_area(self, dataframe):
        current_area = dataframe['fips'].iloc[0]
        dataframe_len = len(dataframe)
        last_domain_index = 0
        for i in range(1,dataframe_len):
            this_area = dataframe['fips'].iloc[i]
            if this_area != current_area:
                self.domain_dict[current_area] = [last_domain_index, i]
                last_domain_index = i
                current_area = this_area
            if i == (dataframe_len-1):
                self.domain_dict[current_area] = [last_domain_index, i+1]
        return self.domain_dict
    #
    #   return the dataset which contains the N (x, y) tensor pairs
    #   from domain I
    #   start index : the start index of domain i
    #   end index : start index + N
    #
    def extract_data(self, dataframe, index):
        column_index = dataframe.columns.get_loc('date')
        row = dataframe.iloc[index]
        current_flag = row['score']
        # remove label
        arr_x = row[:-1].values
        # set time from string to int in datetime
        arr_x[column_index] = datetime.fromisoformat(arr_x[column_index]).timestamp()
        # set type
        numpy_x = arr_x.astype(numpy.float32)
        # change to tensor
        x_tensor = torch.tensor(numpy_x)
        #self.input_shape = x_tensor.shape
        x = x_tensor.unsqueeze(0)
        if not numpy.isnan(current_flag):
            arr_y = row[-1]
            y = torch.tensor(arr_y).long()
            y = y.unsqueeze(0)
            return x, y, current_flag
        else:
            return  x, 9, current_flag



    def orgnize_dataset(self, dataframe, domain_dict, hparams):
        domains = [k for k in domain_dict.keys()]
        self.ENVIRONMENTS = domains
        #assert len(domains) > hparams['env_number']
        #
        total_envs = len(domains)
        num_env = hparams['env_number']
        # how many orignial domains in each domains
        domain_interval = int(total_envs/num_env)
        #
        #ratio = hparams['env_sample_ratio']
        num = hparams['env_sample_number']
        sample_each_class =  int(num/self.num_classes)
        #
        # how many elements in one domain
        #
        domain_sum = {}
        for k1 in self.domain_dict:
            domain = self.domain_dict[k1]
            sum_ = 0
            for k2 in domain:
                sum_ = sum_ + domain[k2][2]
            domain_sum[k1] = sum_
        domain_elements = []
        for i in range(int(num_env)):
            domain_class = {}
            for j in range(self.num_classes):
                domain_class[j] = 0
            start_index = i * domain_interval
            end_index = (i+1) * domain_interval
            #
            # calculate how many elements in one hparams['env_number']
            #
            domain_index_sum = 0
            sum_index = 0
            for key in domain_sum:
                if sum_index in range(start_index, end_index):
                    domain_index_sum = domain_index_sum + domain_sum[key]
                sum_index = sum_index + 1
            domain_elements.append(domain_index_sum)
            # set up index
            # hold the index
            index = sum(domain_elements[:-1])
            x_list = []
            y_list = []
            x_ = []
            # start gathering x, y
            print(" domain i started ", i)
            while(True):
                #print("  index  ", index)
                x1, y, flag = self.extract_data(dataframe, index)
                if y == 9:
                    x_.append(x1)
                    index = index + 1
                    continue
                elif domain_class[y.item()] >= sample_each_class and min(list(domain_class.values())) < sample_each_class:
                    index = index + 1
                    continue
                elif domain_class[y.item()] < sample_each_class and not numpy.isnan(flag):
                    # make the tensor as a 7 days data
                    if len(x_) == 0:
                        x_.append(x1)
                        print(" index ", index)
                    if len(x_) == 7:
                        x = torch.cat(x_, dim = 1)
                    elif len(x_) < 7 :
                        x = torch.cat(x_, dim=1)
                        x_pad = torch.zeros((140 - x.shape[1]))
                        x_pad = x_pad.unsqueeze(0)
                        x = torch.cat((x, x_pad), dim=1)
                    else:
                        leng = len(x_) - 7
                        x = torch.cat(x_[leng:], dim=1)
                    domain_class[y.item()] = domain_class[y.item()] + 1
                    index = index + 1
                    # clear out
                    x_ = []
                    #self.input_shape = x.shape
                    #print(" input shape " ,x.shape)
                    x_list.append(x) # N ( 7*20) => (N, 140)
                    y_list.append(y) # (N, 1)
                    #print(" dictionary is ", domain_class)
                elif min(list(domain_class.values())) == sample_each_class :
                    domain_x = torch.cat(x_list, dim=0) # (N, 7*20)
                    domain_y = torch.cat(y_list, dim=0)
                    domain_dataset = TensorDataset(domain_x, domain_y)
                    self.datasets.append(domain_dataset)
                    print(" x shape tensor ", domain_x.shape)
                    print(" y shape tensor ", domain_y.shape)
                    print("domain ", i, " has been processed ")
                    break
                else:
                    index = index + 1
        return self.datasets


class TDGDroughts(MultipleDomainDataset):

    ENVIRONMENTS = []

    def __init__(self, data_dir, test_envs, hparams):
        super().__init__()
        df = self.load_csv(os.path.join(data_dir, 'train_timeseries.csv'))
        self.datasets = []
        self.domain_dict = {}
        self.num_classes = 6
        hparams['env_number'] = 17
        hparams['env_sample_number'] = 20
        hparams['total_sample_number'] = 0
        hparams['env_sample_ratio'] = 0.5
        self.input_shape = torch.Size([140])
        self.domain_dict = self.orgnize_domain_dict(df)
        self.datasets = self.orgnize_dataset(df, self.domain_dict, hparams)


    def load_csv(self, dir):
        df = pandas.read_csv(dir)
        return df

    # get the domain from year by year
    # return { year :
    #                   { region :
    #                               [ start row index , end_row index (not included), length]} }
    #
    def orgnize_domain_dict(self, dataframe):
        current_year = str(dataframe['date'].iloc[0])[0:4]
        current_area = dataframe['fips'].iloc[0]
        dataframe_len = len(dataframe)
        last_domain_index = 0
        for i in range(1,dataframe_len):
            this_year = str(dataframe['date'].iloc[i])[0:4]
            #print(" this year", this_year, " index ", i)
            this_area = dataframe['fips'].iloc[i]
            if this_year != current_year:
                # from last_domain_index to the i-th, i-th not included
                self.domain_dict[current_year] = {}
                self.domain_dict[current_year][current_area] = [last_domain_index, i, i-last_domain_index]
                last_domain_index = i
                current_year = this_year
                current_area = this_area
                #print(" year  ", current_year)
                #print(" area ", current_area)
                #print(" dict length", len(self.domain_dict))
            if i == (dataframe_len-1):
                self.domain_dict[current_year][current_area] = [last_domain_index, i+1, i+1-last_domain_index]

        return self.domain_dict

    #
    # set the domain as the area
    #
    def orgnize_domain_by_area(self, dataframe):
        current_area = dataframe['fips'].iloc[0]
        dataframe_len = len(dataframe)
        last_domain_index = 0
        for i in range(1,dataframe_len):
            this_area = dataframe['fips'].iloc[i]
            if this_area != current_area:
                self.domain_dict[current_area] = [last_domain_index, i]
                last_domain_index = i
                current_area = this_area
            if i == (dataframe_len-1):
                self.domain_dict[current_area] = [last_domain_index, i+1]
        return self.domain_dict
    #
    #   return the dataset which contains the N (x, y) tensor pairs
    #   from domain I
    #   start index : the start index of domain i
    #   end index : start index + N
    #
    def extract_data(self, dataframe, index):
        column_index = dataframe.columns.get_loc('date')
        row = dataframe.iloc[index]
        current_flag = row['score']
        # remove label
        arr_x = row[:-1].values
        # set time from string to int in datetime
        arr_x[column_index] = datetime.fromisoformat(arr_x[column_index]).timestamp()
        # set type
        numpy_x = arr_x.astype(numpy.float32)
        # change to tensor
        x_tensor = torch.tensor(numpy_x)
        #self.input_shape = x_tensor.shape
        x = x_tensor.unsqueeze(0)
        if not numpy.isnan(current_flag):
            arr_y = row[-1]
            y = torch.tensor(arr_y).long()
            y = y.unsqueeze(0)
            return x, y, current_flag
        else:
            return  x, 9, current_flag



    def orgnize_dataset(self, dataframe, domain_dict, hparams):
        domains = [k for k in domain_dict.keys()]
        self.ENVIRONMENTS = domains
        #assert len(domains) > hparams['env_number']
        #
        total_envs = len(domains)
        num_env = hparams['env_number']
        # how many orignial domains in each domains
        domain_interval = int(total_envs/num_env)
        #
        #ratio = hparams['env_sample_ratio']
        num = hparams['env_sample_number']
        sample_each_class =  int(num/self.num_classes)
        #
        # how many elements in one domain
        #
        domain_sum = {}
        for k1 in self.domain_dict:
            domain = self.domain_dict[k1]
            sum_ = 0
            for k2 in domain:
                sum_ = sum_ + domain[k2][2]
            domain_sum[k1] = sum_
        domain_elements = []
        for i in range(int(num_env)):
            domain_class = {}
            for j in range(self.num_classes):
                domain_class[j] = 0
            start_index = i * domain_interval
            end_index = (i+1) * domain_interval
            #
            # calculate how many elements in one hparams['env_number']
            #
            domain_index_sum = 0
            sum_index = 0
            for key in domain_sum:
                if sum_index in range(start_index, end_index):
                    domain_index_sum = domain_index_sum + domain_sum[key]
                sum_index = sum_index + 1
            domain_elements.append(domain_index_sum)
            # set up index
            # hold the index
            index = sum(domain_elements[:-1])
            x_list = []
            y_list = []
            x_ = []
            # start gathering x, y
            print(" domain i started ", i)
            while(True):
                #print("  index  ", index)
                x1, y, flag = self.extract_data(dataframe, index)
                if y == 9:
                    x_.append(x1)
                    index = index + 1
                    continue
                elif domain_class[y.item()] >= sample_each_class and min(list(domain_class.values())) < sample_each_class:
                    index = index + 1
                    continue
                elif domain_class[y.item()] < sample_each_class and not numpy.isnan(flag):
                    # make the tensor as a 7 days data
                    if len(x_) == 0:
                        x_.append(x1)
                        print(" index ", index)
                    if len(x_) == 7:
                        x = torch.cat(x_, dim = 1)
                    elif len(x_) < 7 :
                        x = torch.cat(x_, dim=1)
                        x_pad = torch.zeros((140 - x.shape[1]))
                        x_pad = x_pad.unsqueeze(0)
                        x = torch.cat((x, x_pad), dim=1)
                    else:
                        leng = len(x_) - 7
                        x = torch.cat(x_[leng:], dim=1)
                    domain_class[y.item()] = domain_class[y.item()] + 1
                    index = index + 1
                    # clear out
                    x_ = []
                    #self.input_shape = x.shape
                    #print(" input shape " ,x.shape)
                    x_list.append(x) # N ( 7*20) => (N, 140)
                    y_list.append(y) # (N, 1)
                    #print(" dictionary is ", domain_class)
                elif min(list(domain_class.values())) == sample_each_class :
                    domain_x = torch.cat(x_list, dim=0) # (N, 7*20)
                    domain_y = torch.cat(y_list, dim=0)
                    domain_dataset = TensorDataset(domain_x, domain_y)
                    self.datasets.append(domain_dataset)
                    print(" x shape tensor ", domain_x.shape)
                    print(" y shape tensor ", domain_y.shape)
                    print("domain ", i, " has been processed ")
                    break
                else:
                    index = index + 1
        return self.datasets
    

class BalancedClassDataset(torch.utils.data.IterableDataset):
    """ BalancedClassDataset
        package data by classes, 
        then each class becomes a TensorDataset.
        At each __iter__, 
        we sample N_p samples from each class. each one with shape (N_p, 1, 28, 28)
        then concat all classes and return
    """

    def __init__(self, x, y, num_classes, N_p):  # TODO add para num_classes
        super(BalancedClassDataset).__init__()
        data_num = x.shape[0]
        dataset = TensorDataset(x, y)
        self.dataloaders = [None for _ in range(num_classes)]
        for each_c in range(num_classes):
            mask = [1 if y[i] == each_c else 0 for i in range(data_num)]
            c_idxs = torch.nonzero(torch.tensor(mask)).flatten()
            sampler = torch.utils.data.SubsetRandomSampler(
                c_idxs, generator=None)
            # see also: torch.utils.data.Subset or torch.utils.data.SubsetRandomSampler(indices, generator=None)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=N_p, sampler=sampler, shuffle=False, num_workers=0)
            self.dataloaders[each_c] = dataloader

    def __iter__(self):
        while True:
            x, y = zip(*[next(iter(dataloader))
                       for dataloader in self.dataloaders])
            x = torch.stack(x, dim=0)
            y = torch.stack(y, dim=0)
            yield (x, y)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3",
                    "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs,
                         hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


def gen_gaussian_alpha_xy_pair(mean, cov, alpha, num):
    """Generate (data_X, data_Y) pairs.

    size = num
    data_X is from a gaussian distribution with mean, cov (two dimension)
    data_Y is calculated as data_X * alpha
    """
    data_X = np.random.multivariate_normal(mean, cov, num)
    alpha = np.array(alpha)
    data_Y = np.matmul(data_X, alpha.reshape((-1, 1))).reshape((num)) # # X*alpha^T aka (num, 2) .mul (2, 1)
    data_Y = np.sign(data_Y)*0.5+0.5 # {-1, 1} -> {0, 1}
    return (data_X, data_Y)

def gen_RPlate():
    """Generate RPlate dataset.

    RPlate dataset consists DOMAIN_NUM where there are SAMPLE_NUM_EACH_DOMAIN
    samples in each domain. P(Y|X) evolves across all domains while P(X) stays
    the same. All P(X) is generated from the same Gaussian distribution with
    MEAN and COV. The classification boundary rotate across domains.

    Output: 
        {
            'data': [], # each element is for one domain 
                # (SAMPLE_NUM_EACH_DOMAIN, 2)
            'label': [], # each element is for one domain 
                # (SAMPLE_NUM_EACH_DOMAIN)
            'domain': [] # [[0, 0, ..., 0], [1, 1, ..., 1], ..., [M, M, ..., M]] 
                # (SAMPLE_NUM_EACH_DOMAIN)
        }
        ps. this format is same with the one from dataset where only P(X) evolves.

    """
    TAR_PATH = "../datasets/RPlate/data/RPlate.pkl"
    DOMAIN_NUM = 30
    SAMPLE_NUM_EACH_DOMAIN = 200
    MEAN = [0, 0]
    COV = [
        [1, 0],
        [0, 1]
    ]
    angle_list = [2*np.pi*i/DOMAIN_NUM for i in range(DOMAIN_NUM)]
    alpha_list = [(np.cos(angle_list[i]), np.sin(angle_list[i])) for i in range(DOMAIN_NUM)]
    res = {
        'data': [],
        'label': [],
        'domain': []
    }
    for d_i  in range(DOMAIN_NUM):
        data_X, data_Y = gen_gaussian_alpha_xy_pair(MEAN, COV, alpha_list[d_i], SAMPLE_NUM_EACH_DOMAIN)
        res['data'].append(data_X)
        res['label'].append(data_Y)
        res['domain'].append(np.array([d_i for _ in range(SAMPLE_NUM_EACH_DOMAIN)]))
    for k, v in res.items():
        res[k] = np.concatenate(v)
    with open(TAR_PATH, 'wb') as f:
        pickle.dump(res, f)
    return res

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# PACS ########################################################################

def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # download or generate dataset
    # gen_RPlate()
    # download_mnist(args.data_dir)
    # FMoWDataset(root_dir=args.data_dir, download=True)