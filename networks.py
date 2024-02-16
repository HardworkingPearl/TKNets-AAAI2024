# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from abc import abstractmethod
import numpy as np

from lib import misc
from lib import wide_resnet


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class Trans(nn.Module):
    """MLP without relu, dropout"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(Trans, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-1)])
        self.gtype = hparams['gtype']
        self.n_outputs = n_outputs

    def forward(self, x):
        if self.gtype == 'fixed':
            x = x + 0.01
            return x
        elif self.gtype == 'same':
            return x
        elif self.gtype in ['linear', 'mlp']:
            x = self.input(x)
            for hidden in self.hiddens:
                x = hidden(x)
            return x
        else:
            raise NotImplementedError

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x

class MNIST_CNN_Decoder(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    # n_outputs = 128

    def __init__(self, input_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, output_shape[0], kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)
        out = self.de1(out)
        out = self.up2(out)
        out = self.de2(out)
        out = self.de3(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], *self.output_shape)
        return out

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


def Featurizer(output_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # print(hparams)
    if 'pure_liner' in hparams:
        return MLP_Empty(output_shape[0], 128, hparams)
    elif len(output_shape) == 1:
        return MLP(output_shape[0], hparams["mlp_width"], hparams)
    elif output_shape[1:3] == (28, 28):
        return MNIST_CNN(output_shape)
    elif output_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(output_shape, 16, 2, 0.)
    elif output_shape[1:3] == (224, 224):
        return ResNet(output_shape, hparams)
    else:
        raise NotImplementedError

class MLP_Empty(nn.Module):
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP_Empty, self).__init__()
        self.n_outputs = n_inputs
        self.tmp = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, x):
        x = x + self.tmp
        return x

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

def DClassifier(in_features, out_features, is_nonlinear=False, num_domains=None):
    if is_nonlinear:
        if in_features == 2:
            if num_domains == 10: # 11:
                return torch.nn.Linear(in_features, out_features)
            else:
                return torch.nn.Sequential(
                    torch.nn.Linear(in_features, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, out_features))
        else:
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

    """
    DIVA module
    Use for reconstructing domain_label and class label
    Adjust from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    """

    def __init__(self, input_dim, output_dim, stochastic=True, init_type='xavier'):
        super(BranchDecoder, self).__init__(output_dim, stochastic)
        if stochastic:
            output_dim = output_dim * 2

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *block(input_dim, output_dim, normalize=True),
            nn.Linear(output_dim, output_dim),
            # nn.Softplus()
        )

    def forward(self, latent_variables):
        out = self.model(latent_variables)
        if self.stochastic:
            self.latent_space = self.gaussian_module(out)
        return out