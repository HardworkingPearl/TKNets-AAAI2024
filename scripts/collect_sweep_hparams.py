

import collections

import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys


sys.path.append("..")
sys.path.append(".")

import numpy as np
from torch.utils import data
import tqdm
import pandas as pd

import datasets
import algorithms
from lib import misc, reporting
import model_selection
from lib.query import Q
import warnings


class RecordsManager:
	def __init__(self, input_dir):
		self.records = reporting.load_records(input_dir)
	
	def to_variations_csv(self, csv_path):
		GROUP_KEYS = ["algorithm", "dataset", "env_distance", "env_number", "env_sample_number", "env_sample_ratio"]
		# get all records as a queue
		records = self.records
		print(f"Total records count (step): {len(records)}")
		# filter to last step
		records = records.filter(lambda g: g['step'] == 5000)
		print(f"Total records count (last step): {len(records)}")

		# group by same as seeds different
		result = collections.defaultdict(lambda: [])
		for r in records:
			group = tuple([
				r["args"][k] for k in GROUP_KEYS
			])
			result[group].append(r)
		records = Q([
				{
					**{GROUP_KEYS[i]:g[i] for i in range(len(GROUP_KEYS))},
					"records": v
				}
			for g, v in result.items()
		])
		print(f"Total records count (different exps): {len(records)}")

		# calculate mean variance and append to group as new key
		records = records.map(lambda group:
			{ 
				**group,
				"query_acc_mean": np.mean([r['query_acc'] for r in group['records']]),
				# "support_acc_mean": np.mean([r['support_acc'] for r in group['records']]),
				"query_acc_var": np.var([r['query_acc'] for r in group['records']]),
				# "support_acc_var": np.var([r['support_acc'] for r in group['records']]),
			})
		# save to pandas and save
		records = records.map(lambda group:
			{
				**group,
				"query_acc_mean": np.mean([r['query_acc'] for r in group['records']]),
				# "support_acc_mean": np.mean([r['support_acc'] for r in group['records']]),
				"query_acc_var": np.var([r['query_acc'] for r in group['records']]),
				# "support_acc_var": np.var([r['support_acc'] for r in group['records']]),
			}
		)
		for i in range(len(records)):
			del records[i]['records']
		df = pd.DataFrame(records._list)
		df.to_csv(csv_path)
		print(f"Save variation csv to: {csv_path}")

	def to_best_hps(self, path):
		GROUP_KEYS = ["algorithm", "dataset", "hparams_seed"]
		# get all records as a queue
		records = self.records
		print(f"Total records count (step): {len(records)}")
		# filter to last step
		records = records.filter(lambda g: g['step'] == 5000)
		print(f"Total records count (last step): {len(records)}")
		# group to only seed is different
		result = collections.defaultdict(lambda: [])
		for r in records:
			group = tuple([
				r["args"][k] for k in GROUP_KEYS
			])
			result[group].append(r)
		records = Q([
				{
					**{GROUP_KEYS[i]:g[i] for i in range(len(GROUP_KEYS))},
					"records": v
				}
			for g, v in result.items()
		])
		print(f"Total records count (different exps+hparams): {len(records)}") # TODO
		# calculate mean
		records = records.map(lambda group:
		{ 
			**group,
			"hparams": group['records'][0]['hparams'],
			"query_acc_mean": np.mean([r['query_acc'] for r in group['records']]),
			# "support_acc_mean": np.mean([r['support_acc'] for r in group['records']]),
			"query_acc_var": np.var([r['query_acc'] for r in group['records']]),
			# "support_acc_var": np.var([r['support_acc'] for r in group['records']]),
		})
		# group [dataset, algorithm]
		group_keys = ['algorithm', 'dataset']
		result = collections.defaultdict(lambda: [])
		for r in records:
			group = tuple([
				r[k] for k in group_keys
			])
			result[group].append(r)
		records = Q([
				{
					**{group_keys[i]:g[i] for i in range(len(group_keys))},
					"records": v
				}
			for g, v in result.items()
		])
		print(f"Total records count (different exps): {len(records)}")
		# find best for each group, append acc + hps
		records = records.map(lambda group:
		{ 
			**group,
			"best_hparams": max(group['records'], key=lambda x:x['query_acc_mean'])['hparams'],
			"best_acc": max(group['records'], key=lambda x:x['query_acc_mean'])['query_acc_mean'],
		})
		# keep some information then save
		records = records.map(lambda group:
			{i:group[i] for i in group if i!='records'}
		)
		with open(path, 'w') as f:
			[f.write(json.dumps(r, sort_keys=True) + "\n") for r in records]
		print(f"Save best hparams to: {path}")
		return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sweep dataset hparams')
	parser.add_argument('--exps_name', type=str)
	parser.add_argument('--type', type=str, default='csv',choices=['csv', 'best_hp'],)
	args = parser.parse_args()
	# config
	INPUT_DIR = f'EXPS/{args.exps_name}'

	# INPUT_DIR = 'EXPS/SimpleDatasetHpSearch'
	OUTPUT_PATH = os.path.join(INPUT_DIR, 'variations_acc.csv')
	# OUTPUT_PATH = os.path.join(INPUT_DIR, 'best_hparams.jsonl')

	record_mgr = RecordsManager(INPUT_DIR)
	if args.type == 'csv':
		record_mgr.to_variations_csv(OUTPUT_PATH)
	elif args.type == 'best_hp':
		record_mgr.to_best_hps(OUTPUT_PATH)
	else:
		raise NotImplementedError
	

	