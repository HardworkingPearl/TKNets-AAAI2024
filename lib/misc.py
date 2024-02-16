# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile

import numpy as np
import torch
import tqdm
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F


def make_weights_for_balanced_classes(dataset):
	counts = Counter()
	classes = []
	for _, y in dataset:
		y = int(y)
		counts[y] += 1
		classes.append(y)

	n_classes = len(counts)

	weight_per_class = {}
	for y in counts:
		weight_per_class[y] = 1 / (counts[y] * n_classes)

	weights = torch.zeros(len(dataset))
	for i, y in enumerate(classes):
		weights[i] = weight_per_class[int(y)]

	return weights


def pdb():
	sys.stdout = sys.__stdout__
	import pdb
	print("Launching PDB, enter 'n' to step to parent function.")
	pdb.set_trace()


def seed_hash(*args):
	"""
	Derive an integer hash from all args, for use as a random seed.
	"""
	args_str = str(args)
	return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
	print("="*80)


def print_row(row, colwidth=10, latex=False):
	if latex:
		sep = " & "
		end_ = "\\\\"
	else:
		sep = "  "
		end_ = ""

	def format_val(x):
		if np.issubdtype(type(x), np.floating):
			x = "{:.10f}".format(x)
		return str(x).ljust(colwidth)[:colwidth]
	print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
	"""Used by split_dataset"""

	def __init__(self, underlying_dataset, keys):
		super(_SplitDataset, self).__init__()
		self.underlying_dataset = underlying_dataset
		self.keys = keys

	def __getitem__(self, key):
		return self.underlying_dataset[self.keys[key]]

	def __len__(self):
		return len(self.keys)


def split_dataset(dataset, n, seed=0):
	"""
	Return a pair of datasets corresponding to a random split of the given
	dataset, with n datapoints in the first dataset and the rest in the last,
	using the given random seed
	"""
	assert(n <= len(dataset))
	keys = list(range(len(dataset)))
	np.random.RandomState(seed).shuffle(keys)
	keys_1 = keys[:n]
	keys_2 = keys[n:]
	return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
	perm = torch.randperm(len(minibatches)).tolist()
	pairs = []

	for i in range(len(minibatches)):
		j = i + 1 if i < (len(minibatches) - 1) else 0

		xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
		xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

		min_n = min(len(xi), len(xj))

		pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

	return pairs


def sequence_pairs_of_minibatches(minibatches, direction="forward_val"):
	"""
	turn minibatches into pairs.
	((x_i, y_i), (x_i+1, y_i+1))
	"""
	pairs = []
	if "forward" in direction:
		for i in range(len(minibatches)):
			if i == len(minibatches) - 1:
				continue
			j = i + 1 if i < (len(minibatches) - 1) else 0

			xi, yi = minibatches[i][0], minibatches[i][1]
			xj, yj = minibatches[j][0], minibatches[j][1]

			min_n = min(len(xi), len(xj))

			pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

	elif "backward" in direction:
		for i_ in range(len(minibatches)):
			i = len(minibatches) - i_ - 1
			if i == 0:
				continue
			j = i - 1

			xi, yi = minibatches[i][0], minibatches[i][1]
			xj, yj = minibatches[j][0], minibatches[j][1]

			min_n = min(len(xi), len(xj))

			pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
	else:
		raise NotImplementedError
	return pairs


def sequence_pairs_of_minibatches_2step(minibatches, direction="forward_val"):
	"""
	turn minibatches into pairs.
	((x_i, y_i), (x_i+1, y_i+1))
	"""
	pairs = []
	if "forward" in direction:
		for i in range(len(minibatches)):
			if i >= len(minibatches) - 2:
				continue
			j = i + 2 if i < (len(minibatches) - 2) else 0

			xi, yi = minibatches[i][0], minibatches[i][1]
			xj, yj = minibatches[j][0], minibatches[j][1]

			min_n = min(len(xi), len(xj))

			pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

	elif "backward" in direction:
		for i_ in range(len(minibatches)):
			i = len(minibatches) - i_ - 1
			if i <= 1:
				continue
			j = i - 2

			xi, yi = minibatches[i][0], minibatches[i][1]
			xj, yj = minibatches[j][0], minibatches[j][1]

			min_n = min(len(xi), len(xj))

			pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
	else:
		raise NotImplementedError
	return pairs


def accuracy_report(algorithm, eval_loaders, eval_weights, test_envs, test_type, device):
	report = {}
	if "TDG" in type(algorithm).__name__:
		for i in range(len(eval_loaders)):
			if "backward" in test_type: support_idx = i+1
			if "forward" in test_type: support_idx = i-1
			if support_idx < 0 or support_idx >= len(eval_loaders): acc = 0.0
			else: acc = tdg_accuracy(algorithm, [eval_loaders[support_idx]], eval_loaders[i], device)
			report["{:03d}".format(i)+("-tr" if i not in test_envs else "-te")] = acc
	else:
		for i in range(len(eval_loaders)):
			acc = accuracy(algorithm, eval_loaders[i], eval_weights[i], device)
			report["{:03d}".format(i)+("-tr" if i not in test_envs else "-te")] = acc
	return report



def accuracy(network, loader, weights, device):
	correct = 0
	total = 0
	weights_offset = 0

	network.eval()
	with torch.no_grad():
		for x, y in loader[0]:
			x = x.to(device)
			y = y.to(device)
			p = network.predict(x)
			if weights is None:
				batch_weights = torch.ones(len(x))
			else:
				batch_weights = weights[weights_offset:
					weights_offset + len(x)]
				weights_offset += len(x)
			batch_weights = batch_weights.to(device)
			if p.size(1) == 1:
				correct += (p.gt(0).eq(y).float() *
							batch_weights.view(-1, 1)).sum().item()
			else:
				correct += (p.argmax(1).eq(y).float() *
							batch_weights).sum().item()
			total += batch_weights.sum().item()
	network.train()

	return correct / total

def dda_accuracy(network, loader, weights, device, domain_idx):
	correct = 0
	total = 0
	weights_offset = 0

	network.eval()
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			p = network.predict(x, domain_idx)
			if weights is None:
				batch_weights = torch.ones(len(x))
			else:
				batch_weights = weights[weights_offset:
										weights_offset + len(x)]
				weights_offset += len(x)
			batch_weights = batch_weights.to(device)
			if p.size(1) == 1:
				correct += (p.gt(0).eq(y).float() *
							batch_weights.view(-1, 1)).sum().item()
			else:
				correct += (p.argmax(1).eq(y).float() *
							batch_weights).sum().item()
			total += batch_weights.sum().item()
	network.train()

	return correct / total


def tdg_accuracy(algorithm, support_loaders, query_loader, device, args=None):
	correct = 0
	total = 0

	algorithm.eval_setup(support_loaders, device, args)  # setup with support

	with torch.no_grad():  # eval with query
		for x, y in query_loader:
			x = x.to(device)
			y = y.to(device)
			p = algorithm.eval_predict(x)
			if p.size(1) == 1:
				correct += (p.gt(0).eq(y).float()).sum().item()
			else:
				correct += (p.argmax(1).eq(y).float()).sum().item()
			total += x.shape[0]
	algorithm.train()

	return correct / total


class MMD_loss(nn.Module):
	def __init__(self, kernel_mul=2.0, kernel_num=5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return

	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)

		total0 = total.unsqueeze(0).expand(
			int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(
			int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i)
						  for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
					  for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(
			source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY - YX)
		return loss


class Tee:
	def __init__(self, fname, mode="a"):
		self.stdout = sys.stdout
		self.file = open(fname, mode)

	def write(self, message):
		self.stdout.write(message)
		self.file.write(message)
		self.flush()

	def flush(self):
		self.stdout.flush()
		self.file.flush()


"""LSSAE"""


def one_hot(indices, depth, device=None):
	"""
	Returns a one-hot tensor.
	This is a PyTorch equivalent of Tensorflow's tf.one_hot.
	Parameters:
	  indices:  a (n_batch, m) Tensor or (m) Tensor.
	  depth: a scalar. Represents the depth of the one hot dimension.
	Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
	"""
	if device is None:
		encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
	else:
		encoded_indicies = torch.zeros(
			indices.size() + torch.Size([depth])).to(device)

	index = indices.view(indices.size() + torch.Size([1]))
	encoded_indicies = encoded_indicies.scatter_(1, index, 1)

	return encoded_indicies


def kl_divergence(latent_space_a, latent_space_b):
	return torch.mean(torch.distributions.kl_divergence(latent_space_a, latent_space_b))


def temporal_smooth_loss(latent_variables, batch_first=True):
	if batch_first:
		return F.l1_loss(latent_variables[:, 1:, :], latent_variables[:, :-1, :], reduction='mean')
	else:
		return F.l1_loss(latent_variables[1:, :, :], latent_variables[-1:, :, :], reduction='mean')
