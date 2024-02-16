# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np

def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples. sort with val_acc
        return the [({'query_acc': 0.925, ...}, [records, ...]), ...] with each element as one hparams_seed
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records: # two records 
                (
                    self.run_acc(run_records), # dict {'query_acc': 0.925, 'val_acc': 0.925, 'test_acc': 0.925}
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc'] # sort with val_acc then return the test acc (single float)
        else:
            return None
    
    @classmethod
    def top_hparams(self, records):
        """ TODO
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][1][0] # list, (acc, records), records_list
        else:
            return None
        

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None


class TDGMethod(SelectionMethod):
    name = "TDG max query acc"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {query_acc} dict
        """
        return {
            'query_acc': record['query_acc'],
            'val_acc': record['query_acc'], # same as upper for sort 
            'test_acc': record['query_acc'], # TODO we need to calculate test and val seperately here, sort by query but return val
        }

    @classmethod
    def run_acc(self, run_records):
        # test_records = get_test_records(run_records)
        test_records = run_records # TODO add filter "test" in test_type
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('query_acc')

class TDGMultiStepMethod(SelectionMethod):
    name = "TDG max query acc"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {query_acc} dict
        """
        return {
            'query_acc': record['003-te'], # for choose best step in one run
            'val_acc': record['004-tr'], # for sorting for different hparams
            # 'test_acc': record['007-tr'], # 
            'test_acc': record['006-tr'], # 
            # 'test_acc': record['005-tr'], # 
            # 'test_acc': record['004-tr'], # 
            # 'test_acc': record['003-te'], # for report
            # 'test_acc': record['002-te'], # 
            # 'test_acc': record['001-te'], # 
            # 'test_acc': record['000-te'], # 
            'step_1': record['003-te'], # 
            'step_2': record['002-te'], # 
            'step_3': record['001-te'], # 
            'step_4': record['000-te'], # 
            'step': record['step']
        }

    @classmethod
    def run_acc(self, run_records):
        """return the record with highest querry_acc"""
        # test_records = get_test_records(run_records)
        test_records = run_records # TODO add filter "test" in test_type
        if not len(test_records):
            return None
        # run_records.sorted(lambda r: r['step'])[-1]
        # return test_records.map(self._step_acc).argmax('query_acc') # choose best from 50 steps in one run
        # return test_records.map(self._step_acc).argmax('step')
        return test_records.map(self._step_acc).argmax('val_acc')
