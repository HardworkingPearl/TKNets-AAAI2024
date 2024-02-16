# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import itertools
from torch.autograd import Function
from torch.autograd import Variable

import copy
import numpy as np
from tqdm import tqdm
from abc import abstractmethod

import networks
from networks import *
from lib.misc import (
    random_pairs_of_minibatches,
    sequence_pairs_of_minibatches, sequence_pairs_of_minibatches_2step, MMD_loss, one_hot, temporal_smooth_loss
)

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'TDG',
    'TKNets'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        self.num_domains = num_domains

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def eval_setup(self, support_loaders, device, args):
        pass

    def eval_predict(self, xq):
        return self.network(xq)

    def update(self, minibatches, unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        # loss = F.cross_entropy(self.predict(all_x), all_y)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        total_loss = 0
        for x, y in minibatches:
            loss = F.cross_entropy(self.predict(x), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {'loss': total_loss}

    def predict(self, x):
        return self.network(x)

    def embedding(self, x):
        return self.featurizer(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape(
            (meta_batch_size, support_size, 1, h, w))  # TODO what is this?
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(
                disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def eval_setup(self, support_loaders, device, args):
        pass

    
    def eval_predict(self, xq):
        return self.classifier(self.featurizer(xq))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
                                    hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                       hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class TDG(ERM):
    SEQUENCE_PAIR = False  # use sequential pairs, False for ramdom pairs
    UPGRADED_PROTO = False  # use double network

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TDG, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        hparams['dmatch_rate'] = hparams['dmatch_rate'] if 'dmatch_rate' in hparams else 0.
        self.num_classes = num_classes
        self.encoder_s = networks.Featurizer(input_shape, self.hparams)
        self.encoder_q = networks.Featurizer(input_shape, self.hparams)
        self.support_center = None  # in frozen mode, the predict will use this support center
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.encoder_s.parameters(),
                            self.encoder_q.parameters()),  
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def predict(self, x):
        '''use support as center, then return the class'''
        # TODO
        return

    def cal_loss_main(self, xs, ys, xq, yq):
        zs = self.embedding_s(xs)  # [batchsize, z_dim]
        zq = self.embedding_q(xq)  # [batchsize, z_dim]
        res = self.cal_center(zs, ys)  # [n_class, z_dim]
        if type(res) == bool:
            return False, False
        z_proto = res
        dists =  self.euclidean_dist(zq, z_proto)  # [batchsize, n_class]
                 # self.cosine_dist(zq, z_proto) #
        log_p_y = F.log_softmax(-dists, dim=1)  # [batchsize, n_class]
        loss_main = -log_p_y.gather(1, yq.unsqueeze(1)
                                    ).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, yq).float().mean()
        return loss_main, acc

    def cal_loss_dmatch(self, xs, ys, sq, yq):
        return torch.tensor(0.0)

    def update(self, minibatches, unlabeled=None):
        pair_func = sequence_pairs_of_minibatches if self.SEQUENCE_PAIR else lambda batches, hp: random_pairs_of_minibatches(
            batches)
        loss_sum = 0
        loss_mains = []
        loss_dmatchs = []
        loss_totals = []
        accs = []
        for (xs, ys), (xq, yq) in pair_func(minibatches, self.hparams['test_type']):
            loss_main, acc = self.cal_loss_main(xs, ys, xq, yq)
            if not loss_main:
                continue  # for case of none support
            # TODO function self.cal_loss_main(xs, ys, xq, yq)
            loss_dmatch = self.cal_loss_dmatch(xs, ys, xq, yq)
            loss_total = loss_main + self.hparams['dmatch_rate'] * loss_dmatch
            loss_mains.append(loss_main.item())  # log
            loss_dmatchs.append(loss_dmatch.item())
            loss_totals.append(loss_total.item())
            accs.append(acc.item())
            loss_sum+=loss_total
        self.optimizer.zero_grad()  # backward
        loss_sum.backward()
        self.optimizer.step()

        return {
            'loss_main': np.mean(loss_mains),
            'loss_dmatch': np.mean(loss_dmatchs),
            'loss_total': np.mean(loss_totals),
            'acc': np.mean(accs)
        }

    def cal_cond_mmd(self, xs, ys, xq, yq):
        '''
        calculate total mmd loss for samples of each classes
        '''
        mmd_losses = []
        for c_i in range(self.num_classes):
            s_idxes = torch.nonzero(torch.eq(ys, c_i)).flatten()
            q_idxes = torch.nonzero(torch.eq(yq, c_i)).flatten()
            len_crop = min(s_idxes.shape[0], q_idxes.shape[0])
            s_idxes, q_idxes = s_idxes[:len_crop], q_idxes[:len_crop]
            if len_crop == 0:
                break
            mmd_losses.append(MMD_loss()(xs[s_idxes], xq[q_idxes]))
        mmd_loss = torch.sum(torch.tensor(mmd_losses))
        return mmd_loss

    def cal_center(self, zs, ys):
        '''
        use output of encoder + label
        to calculate center of each class for latter classification
        '''
        centers = []
        for c_i in range(self.num_classes):
            c_idxes = torch.nonzero(torch.eq(ys, c_i)).flatten()
            # print(f"{c_i}: {c_idxes.shape[0]}")
            if(c_idxes.shape[0] == 0):
                # print("false zero class")
                return False
            ct = zs[c_idxes].mean(0)
            centers.append(ct)
        return torch.stack(centers)

    def cosine_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return - F.cosine_similarity(x, y, dim=2)

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)  # [n, m]

    def eval_setup(self, support_loaders, device, args):
        '''use data in support_loaders to find the support_center[n_class, z_dim]'''
        loader = support_loaders[0]  # TODO using all may be better. here we input all the train loaders but only use the last one
        z_list = []
        y_list = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader, 0):
                x = x.to(device)
                y = y.to(device)
                z = self.embedding_s(x)  # [batchsize, z_dim]
                z.to('cpu')
                y.to('cpu')
                z_list.append(z)
                y_list.append(y)
            # [data_num, z_dim]  
            zs = torch.cat(z_list, dim=0)
            ys = torch.cat(y_list, dim=0)  # [data_num, ]
            z_proto = self.cal_center(zs, ys)  # [n_class, z_dim]
            self.support_center = z_proto
            # print("set self.support_center as: {}".format(z_proto))

    def eval_predict(self, xq):
        zq = self.embedding_q(xq)  # [batchsize, z_dim]
        try:
            dists = self.euclidean_dist( # self.cosine_dist( #
                zq, self.support_center)  # [batchsize, n_class]
        except:
            return torch.zeros(len(xq),self.num_classes).to(xq.device)
        log_p_y = F.log_softmax(-dists, dim=1)  # [batchsize, n_class]
        _, y_hat = log_p_y.max(1)
        return log_p_y  # [batch_size, 1] or [batch_size, class_num]

    def embedding(self, x):
        return self.encoder_s(x)

    def embedding_s(self, x):
        return self.encoder_s(x)

    def embedding_q(self, x):
        return self.encoder_q(x) if self.UPGRADED_PROTO else self.encoder_s(x)



class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                          hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class TKNets(TDG):
    SEQUENCE_PAIR = True  # use sequential pairs, False for ramdom pairs
    UPGRADED_PROTO = True  # use double network
    GTYPE = 'linear'  # linear, mlp, fixed

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains,
                         hparams)
        self.encoder = networks.Featurizer(input_shape, self.hparams)

        if 'gtype' in self.hparams:
            self.hparams['gtype'] = self.GTYPE

        self.G = "pre-define"
        self.trans = networks.Trans(
        self.featurizer.n_outputs,
        self.featurizer.n_outputs, {
            'mlp_width': self.featurizer.n_outputs,
            'mlp_dropout': 0.5,  # not used
            'mlp_depth': 1 if self.GTYPE == 'linear' else 3,
            'gtype': self.GTYPE}
        )
        self.support_center = None  # in frozen mode, the predict will use this support center
        self.optimizer = torch.optim.Adam(
            # itertools.chain(self.encoder.parameters(), self.mlp.parameters(), self.trans.parameters()),
            itertools.chain(self.encoder.parameters(),
                            self.trans.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        

    def g(self, x):
        if self.G == "pre-define":
            d = self.featurizer.n_outputs 
            
            # polynomials
            x_poly1 = x[:, :d//8] ** 2  # 2-th order
            x_poly2 = x[:, d//8:d//4] ** 3  # 3-th order
    
            # exponential function
            x_exp = torch.exp(x[:, d//4:d//2])

            # sine/cos functions
            x_cos = torch.cos(x[:, d//2:d//8*5])
            x_sin = torch.sin(x[:, d//8*5:d//4*3])

            x = torch.cat([x_poly1,x_poly2,x_exp,x_cos,x_sin,x[:,d//4*3:]],dim=-1)
        return x

    def embedding_s(self, x):
        x = self.encoder(x)
        x = self.trans(x)
        return self.g(x)

    def embedding_q(self, x):
        x = self.encoder(x)
        return self.g(x)


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                  num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                    num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                     hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                 hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


"""lssae"""


class AbstractAutoencoder(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.hparams = hparams
        self.featurizer = networks.Featurizer(input_shape, hparams)
        # self.classifier = networks.Classifier( # TODO check
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     self.hparams['nonlinear_classifier'])
        self.classifier = networks.Classifier(
            hparams['zc_dim'] + num_classes,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.model_func = self.featurizer
        self.cla_func = self.classifier
        self.feature_dim = self.model_func.n_outputs
        self.data_size = input_shape  # hparams['data_size']
        self.num_classes = num_classes
        self.seen_domains = num_domains

        self.zc_dim = hparams['zc_dim']
        self.zw_dim = hparams['zw_dim']
        self.zv_dim = num_classes # TODO

        self.recon_criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _get_decoder_func(self):
        if len(self.data_size) > 2:
            if self.data_size[-1] == 28:
                decoder_class = CovDecoder28x28
            elif self.data_size[-1] == 84:
                decoder_class = CovDecoder84x84
            else:
                raise ValueError('Don\'t support shape:{}'.format(
                    self.hparams['data_size']))
        else:
            decoder_class = LinearDecoder
        return decoder_class

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def update(self, minibatches, unlabeled=False):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        pass

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        pass

    def calc_recon_loss(self, recon_x, x):
        recon_loss = self.recon_criterion(recon_x, x)
        recon_loss = recon_loss.sum()
        return recon_loss

    def update_scheduler(self):
        pass


class VAE(AbstractAutoencoder):
    """
    Implementation of Vanilla VAE.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(VAE, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.encoder = ProbabilisticEncoder(
            self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.decoder = self._get_decoder_func()(
            self.hparams['zc_dim'], self.data_size)

        self.domain_cla_func = SingleLayerClassifier(
            self.hparams['zc_dim'], self.seen_domains)

        self.opt = torch.optim.Adam([{'params': self.encoder.parameters()},  #
                                     {'params': self.decoder.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

        self.opt_cla = torch.optim.Adam(params=self.cla_func.parameters(), lr=self.hparams["lr"],
                                        weight_decay=self.hparams['weight_decay'])

        self.opt_domain_cla = torch.optim.Adam(params=self.domain_cla_func.parameters(), lr=self.hparams["lr"],
                                               weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            # Step1: Optimize VAE
            _ = self.encoder(x)
            zx_q = self.encoder.sampling()
            recon_x = self.decoder(zx_q)
            recon_loss = self.calc_recon_loss(recon_x, x)
            recon_kl_loss = kl_divergence(
                self.encoder.latent_space, self.gaussian.latent_space)
            vae_loss = recon_loss + recon_kl_loss
            self.opt.zero_grad()
            vae_loss.backward()
            self.opt.step()

            # Classification
            zx_mu = Variable(
                self.encoder.latent_space.base_dist.loc, requires_grad=True)
            pred_logit = self.cla_func(zx_mu)
            cla_loss = self.criterion(pred_logit, y)
            self.opt_cla.zero_grad()
            cla_loss.backward()
            self.opt_cla.step()

            domain_y = torch.ones_like(y) * domain_idx
            zx_mu = Variable(
                self.encoder.latent_space.base_dist.loc, requires_grad=True)
            domain_logit = self.domain_cla_func(zx_mu)
            domain_cla_loss = self.criterion(domain_logit, domain_y)
            self.opt_domain_cla.zero_grad()
            domain_cla_loss.backward()
            self.opt_domain_cla.step()

            # Append status for each domain
            cla_losses.append(cla_loss)
            all_y_pred.append(pred_logit)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        training = False
        _ = self.encoder(x)
        zx_q = self.encoder.latent_space.base_dist.loc
        output = self.cla_func(zx_q)
        return output

    def reconstruct_for_test(self, x, generative=False):
        with torch.no_grad():
            if generative:
                zx = self.gaussian.sampling(x.size(0))
            else:
                _ = self.encoder(x)
                zx = self.encoder.sampling()
            recon_x = self.decoder(zx)
        return recon_x


class DIVA(AbstractAutoencoder):
    """
    Implementation of DIVA.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(DIVA, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self.aux_loss_multiplier_d = 3500
        self.aux_loss_multiplier_y = 2000
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.qzx = ProbabilisticEncoder(
            self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.qzy = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])
        self.qzd = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])

        self.px = self._get_decoder_func()(
            self.hparams['zc_dim'] + 2 * self.hparams['zdy_dim'], self.data_size)

        self.pzd = BranchDecoder(
            self.seen_domains, self.hparams['zdy_dim'], self.hparams['stochastic'])
        self.pzy = BranchDecoder(
            self.num_classes, self.hparams['zdy_dim'], self.hparams['stochastic'])

        # Auxiliary branch
        self.qd = SingleLayerClassifier(
            self.hparams['zdy_dim'], self.seen_domains)
        self.qy = self.cla_func

        self.opt = torch.optim.Adam([{'params': self.qzx.parameters()},  #
                                     {'params': self.qzy.parameters()},
                                     {'params': self.qzd.parameters()},
                                     {'params': self.px.parameters()},
                                     {'params': self.pzd.parameters()},
                                     {'params': self.pzy.parameters()},
                                     {'params': self.qd.parameters()},
                                     {'params': self.qy.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            domain_y = torch.ones_like(y) * domain_idx

            _ = self.qzx(x)
            _ = self.qzy(x)
            _ = self.qzd(x)

            zx_q = self.qzx.sampling()
            zy_q = self.qzy.sampling()
            zd_q = self.qzd.sampling()

            recon_x = self.px(torch.cat([zd_q, zx_q, zy_q], dim=1))

            one_hot_y = one_hot(y, self.num_classes, x.device)
            one_hot_d = one_hot(domain_y, self.seen_domains, x.device)

            _ = self.pzy(one_hot_y)
            _ = self.pzd(one_hot_d)

            d_hat = self.qd(zd_q)
            y_hat = self.qy(zy_q)

            CE_x = self.calc_recon_loss(recon_x, x)
            zd_p_minus_zd_q = torch.sum(self.pzd.latent_space.log_prob(
                zd_q) - self.qzd.latent_space.log_prob(zd_q))
            KL_zx = torch.sum(self.gaussian.latent_space.log_prob(
                zx_q) - self.qzx.latent_space.log_prob(zx_q))
            zy_p_minus_zy_q = torch.sum(self.pzy.latent_space.log_prob(
                zy_q) - self.qzy.latent_space.log_prob(zy_q))

            # Semantic classification
            CE_y = self.criterion(y_hat, y)
            # Domain classification
            CE_d = self.criterion(d_hat, domain_y)

            loss = CE_x - zd_p_minus_zd_q - KL_zx - zy_p_minus_zy_q + self.aux_loss_multiplier_d * CE_d + \
                self.aux_loss_multiplier_y * CE_y
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Append status for each domain
            cla_losses.append(loss)
            all_y_pred.append(y_hat)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        _ = self.qzy(x)
        zx_q = self.qzy.latent_space.base_dist.loc
        output = self.qy(zx_q)
        return output

    def predict_domain(self, x, *args, **kwargs):
        _ = self.qzd(x)
        zd_q = self.qzd.latent_space.base_dist.loc
        output = self.qd(zd_q)
        return output

    def reconstruct_for_test(self, x, generative=False):
        with torch.no_grad():
            _ = self.encoder(x)
            if generative:
                zx_q = self.gaussian.sampling(x.size(0))
            else:
                zx_q = self.qzx.sampling()
            _ = self.qzy(x)
            _ = self.qzd(x)
            zd_q = self.qzd.sampling()
            zy_q = self.qzy.sampling()
            recon_x = self.px(torch.cat([zd_q, zx_q, zy_q], dim=1))
        return recon_x


