# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.scheduler import CosineLRScheduler

from config import get_config
from models import build_model
from models import build_vmamba, build_vmamba_512, build_vmamba_new
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, jaccard_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from matplotlib import pyplot as plt
from itertools import cycle

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=104, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--output-dir', default='./output/', type=Path,
                    metavar='DIR', help='path to output directory')

import pandas as pd
import time
#import cv2
from PIL import Image
from torchvision import transforms as trans
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, jaccard_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from ASL import AsymmetricLoss

#AMP

import torch
import numpy as np
import math
import torch.distributed as dist
import os

threshold = 0.8

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, dist=False, difficult_example=False, ws=4):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.distributed = dist
        self.difficult_example = difficult_example
        self.world_size = ws
    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage()).cuda()
        self.targets = torch.LongTensor(torch.LongStorage()).cuda()

    def get_results(self):
        return self.targets, self.scores
    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if self.distributed:
            ScoreSet = [torch.zeros_like(self.scores) for _ in range(self.world_size)]
            TargetSet = [torch.zeros_like(self.targets) for _ in range(self.world_size)]

            # print("sc1", self.scores.shape)
            dist.all_gather(ScoreSet,self.scores)
            dist.all_gather(TargetSet,self.targets)

            ScoreSet = torch.cat(ScoreSet)
            TargetSet = torch.cat(TargetSet)
            self.scores = ScoreSet.detach().cpu()
            self.targets = TargetSet.detach().cpu()
        # print("sc2", self.scores.shape)
        # else:
        #     return torch.tensor([0.0])

        # dist.all_reduce_multigpu(self.scores, 0)
        # dist.all_reduce_multigpu(self.targets, 0)
        # print("sc2", self.targets.shape)
        # print(self.t)
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class

        for k in range(self.scores.size(1)):
            # print("k", k)
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            # if self.difficult_example:
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets,self.difficult_example)
            # else:
                # ap[k] = AveragePrecisionMeter.average_precision_coco(scores, targets)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_example):

        # sort examples
        # print("fin")
        # print(output.shape)
        sorted, indices = torch.sort(output, dim=0, descending=True)
        # print(indices)
        # indices = range(len(output))
        # Computes prec@i
        # print(target)
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:

            label = target[i]
            if difficult_example and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        try:
            precision_at_i /= pos_count
        except ZeroDivisionError:
            precision_at_i = 0
        return precision_at_i

    @staticmethod
    def average_precision_coco(output, target):
        epsilon = 1e-8
        output=output.cpu().numpy()
        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i

    def overall(self):

        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        scoring = np.where(scores>=threshold, 1, 0)
        if self.difficult_example:
            targets[targets == -1] = 0
        return self.evaluation(scoring, targets)

    def overall_topk(self, k):
        # print(self.scores, self.targets)
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c))
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        # print(index)
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >=threshold else 0 ### Thersholder!!!
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        # print(scores_)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores == 1)
            Nc[k] = np.sum(targets * (scores == 1))
        # print(np.sum(Nc), np.sum(Np), np.sum(Ng))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)
        OF1 = OF1 if not math.isnan(OF1) else 0

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        CF1 = CF1 if not math.isnan(CF1) else 0
        return OP, OR, OF1, CP, CR, CF1

def on_start_epoch(meter):
    meter['ap_meter'].reset()
    return meter

def on_end_epoch(meter, training, config, epoch=0, distributed=False):
    map = 100 * meter['ap_meter'].value()
    class_map = None
    if  meter['ap_meter'].difficult_example:
        class_map = map
    map = map.mean()
    OP, OR, OF1, CP, CR, CF1 = meter['ap_meter'].overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = meter['ap_meter'].overall_topk(3)
    if distributed:
        local_rank = int(os.environ.get("SLURM_LOCALID")) if config.computecanada else int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
    if not distributed or (local_rank == 0):
        if training:

            print('Epoch: [{0}]\t'
                    'mAP {map:.3f}'.format(epoch, map=map))
            print('OP: {OP:.4f}\t'
                    'OR: {OR:.4f}\t'
                    'OF1: {OF1:.4f}\t'
                    'CP: {CP:.4f}\t'
                    'CR: {CR:.4f}\t'
                    'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        else:

            print('Test: \t mAP {map:.3f}'.format(map=map))
            print('OP: {OP:.4f}\t'
                    'OR: {OR:.4f}\t'
                    'OF1: {OF1:.4f}\t'
                    'CP: {CP:.4f}\t'
                    'CR: {CR:.4f}\t'
                    'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            print('OP_3: {OP:.4f}\t'
                    'OR_3: {OR:.4f}\t'
                    'OF1_3: {OF1:.4f}\t'
                    'CP_3: {CP:.4f}\t'
                    'CR_3: {CR:.4f}\t'
                    'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
    if distributed:
        dist.barrier()

    return {"map": map.numpy(),"class_map":class_map, "OP": OP, "OR": OR, "OF1": OF1, "CP": CP, "CR":CR, "CF1": CF1, "OP_3": OP_k, "OR_3": OR_k, "OF1_3": OF1_k, "CP_3": CP_k, "CR_3": CR_k, "CF1_3":CF1_k} #, meter['ap_meter'].overall()


def on_end_batch(meter,preds, labels ):

    # measure mAP
    meter['ap_meter'].add(preds, labels)
    # print(preds)
    return meter

def initialize_meters(dist, difficult_example, ws):
    meters = {}
    meters['ap_meter'] = AveragePrecisionMeter(dist=dist, difficult_example=difficult_example, ws=ws)


    return meters

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#--LOSS FN--
# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Loss functions for VOLO
import torch
import torch.nn as nn
import torch.nn.functional as F

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class SoftTargetCrossEntropy(nn.Module):
    """
    The native CE loss with soft target
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def compute_and_log_metrics(prefix, labels, outputs, epoch, writer, threshold=0.5):
    precision_micro = precision_score(labels, (outputs > threshold), average='micro', zero_division=0.0)
    recall_micro = recall_score(labels, (outputs > threshold), average='micro', zero_division=0.0)
    f1_micro = f1_score(labels, (outputs > threshold), average='micro', zero_division=0.0)

    precision_macro = precision_score(labels, (outputs > threshold), average='macro', zero_division=0.0)
    recall_macro = recall_score(labels, (outputs > threshold), average='macro', zero_division=0.0)
    f1_macro = f1_score(labels, (outputs > threshold), average='macro', zero_division=0.0)

    jaccard = jaccard_score(labels, (outputs > threshold), average='samples', zero_division=0.0)
    hamming = hamming_loss(labels, (outputs > threshold))
    accuracy = hamming_score(labels, (outputs > threshold))

    ap = average_precision_score(labels, outputs, average=None)
    total = 0
    map = 0
    for score in ap:
      if not np.isnan(score):
        total += 1
        map += score
    map = map / total

    print('mAP is: {}'.format(map))

    writer.add_scalar(f'{prefix}/Precision-micro', precision_micro, epoch)
    writer.add_scalar(f'{prefix}/Recall-micro', recall_micro, epoch)
    writer.add_scalar(f'{prefix}/F1-micro', f1_micro, epoch)

    writer.add_scalar(f'{prefix}/Precision-macro', precision_macro, epoch)
    writer.add_scalar(f'{prefix}/Recall-macro', recall_macro, epoch)
    writer.add_scalar(f'{prefix}/F1-macro', f1_macro, epoch)

    writer.add_scalar(f'{prefix}/Hamming-loss', hamming, epoch)
    writer.add_scalar(f'{prefix}/Jaccard', jaccard, epoch)
    writer.add_scalar(f'{prefix}/Accuracy (Hamming Score)', accuracy, epoch)
    writer.add_scalar(f'{prefix}/mAP', map, epoch)

from torch.cuda.amp import GradScaler, autocast

class MultilabelDataset(torch.utils.data.Dataset):

        def __init__(self, df, image_name_idx=0, label_starting_idx = 1, transforms = None):
            self.df = df
            if transforms:
              self.transforms = transforms
            else:
              self.transforms = trans.Compose([trans.Resize((512, 512)),
                trans.ToTensor()
                ])

            self.name_idx = image_name_idx
            self.start_idx = label_starting_idx

        def __len__(self):
            return len(self.df)

        def __getitem__(self,idx):
            #zeroeth col, idx row
            img = Image.open(self.df.iloc[idx, self.name_idx]).convert("RGB")
            # img_text = self.df.iloc[idx, self.name_idx]
            label = self.df.iloc[idx, self.start_idx:]

            return self.transforms(img), torch.Tensor([label])
            # return img_text, torch.Tensor([label])

#def data_sanity_check(train_df):
#    """
#        this will check each image file for corrupted or missing and
#        returns index of corrupted / missing files .Doing this will
#        prevent us from running into any data errors during training phase .
#    """
#    print('Checking Corrupted Images...')
#    idx = []
#    pbar = tqdm(total=len(train_df))
#    for i in range(len(train_df)):
#        try:#       checks for corrupeted or missing image files
#            if len(cv2.imread(train_df.iloc[i,0])) == 3:
#                _ = 1
#        except:
#            idx.append(i)
#        pbar.update(1)
#    pbar.close()

#    return idx

def prepare_dataset(img_dir_path, csv_path, meta_path, image_name_idx=0, label_starting_idx=2,
                    sanity_check=False, transforms=None, classes = None):
  train_df = pd.read_csv(csv_path)
  train_df["patch_image_id"] = train_df["patch_image_id"].astype(str)

  print('len of df:', len(train_df), "from path:" , csv_path)

  train_df["patch_image_id"] = train_df["patch_image_id"].map(lambda x: os.path.join(img_dir_path, x + '_resized.png'))

  #print('filtered len of train df:', len(train_df))  
  if classes is not None:
    train_df = train_df[['patch_image_id']+ classes]
  #if sanity_check:
  #  corrupted = data_sanity_check(train_df)
  #  train_df = train_df.drop(index=corrupted)

  #take in a df 
  dataset = MultilabelDataset(train_df, image_name_idx, label_starting_idx, transforms=transforms)
  return dataset, train_df

def calculate_class_weights(labels, num_classes):
    label_columns = labels.columns[1:]  # Assumes the first column is not a label column
    
    class_weights = []
    for label in label_columns:
        class_count = labels[label].sum()
        
        # Handling class with zero occurrences
        if class_count == 0:
            # Assign a default high weight to encourage learning when it does occur
            # This default weight can be a subject of tuning based on how you want to treat these cases
            class_weights.append(5.0)  # Example high weight, or you could use np.max(weights) if calculated later
        else:
            # Apply logarithmic scaling to the weight calculation
            weight = np.log(10395 / class_count) + 1
            class_weights.append(weight)
    
    print("Class weights: ", class_weights)
    return torch.tensor(class_weights, dtype=torch.float32)

class FineTunedModel(nn.Module):
    def __init__(self, backbone, num_classes=28):
        super().__init__()
        self.backbone = backbone
        # for param in self.backbone.parameters():
        #    param.requires_grad = False
        # Assuming the output of the backbone is 1024-dimensional as per the last layer before the projector
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        # Pass the output through the new fully connected layer
        return self.fc(x)

import random
import torch

def set_seed_ddp(seed, ddp):
    seed = seed + rank
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    # if args.rank == 0:
    #     args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #     stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    #     print(' '.join(sys.argv))
    #     print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    #info msg
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-6,
        t_in_epochs=True
    )

    # Automatically resume from checkpoint if it exists
    if (args.checkpoint_dir).is_file():
        ckpt = torch.load(args.checkpoint_dir, map_location='cpu')
        start_epoch = ckpt['epoch']
        # model_state_dict = model.state_dict()
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print('Starting from checkpoint:', args.checkpoint_dir)
        print('Finished pretraining at epoch: ', start_epoch)
    else:
        start_epoch = 0    
        
    label_starting_idx = 1
    name_idx = 0

    img_dir = '/home/likai16/scratch/annotated_colon_780'
    # csv_path = '/home/likai16/scratch/colon_annotations_nonzero.csv'
    meta_path = '/home/likai16/projects/rrg-msh/likai16/colon_metadata.csv'

    #new
    # train_path = '/home/likai16/scratch/colon_annotations_train.csv'
    # val_path = '/home/likai16/scratch/colon_annotations_val.csv'
    # test_path = '/home/likai16/scratch/colon_annotations_test.csv'

    train_path = '/home/likai16/projects/def-msh-ab/likai16/train_split.csv'
    val_path = '/home/likai16/projects/def-msh-ab/likai16/val_split.csv'
    test_path = '/home/likai16/projects/def-msh-ab/likai16/test_split.csv'

    checkpoint_save_dir = args.output_dir

    log_dir = checkpoint_save_dir
    #change the transformations instead of randaugment
    train_transform = trans.Compose([trans.Resize((512, 512)),
                trans.RandomHorizontalFlip(p=0.5),
                trans.RandomRotation(degrees=180),
                trans.RandomApply(
                [trans.ColorJitter(brightness=0.2, contrast=0.2,
                                        saturation=0.1, hue=0.05)],
                p=0.8
                ),
                #trans.RandAugment(num_ops = 2, magnitude= 9, num_magnitude_bins=31),
                trans.ToTensor(),
                transforms.Normalize(mean = [0.8710, 0.6560, 0.7565], std = [0.1524, 0.2428, 0.1715]),
                trans.RandomErasing()
                ])
    val_transform = trans.Compose([trans.Resize((512, 512)),
                trans.ToTensor(),
                #trans.Normalize([0.81233799, 0.64032477, 0.81902153], [0.18129702, 0.25731668, 0.16800649])
                #trans.Normalize(mean = [0.8899, 0.7022, 0.7790], std = [0.0985, 0.1793, 0.1416])
                transforms.Normalize(mean = [0.8710, 0.6560, 0.7565], std = [0.1524, 0.2428, 0.1715]),
                ])

    #new
    train_dataset, _ = prepare_dataset(img_dir, train_path, meta_path, image_name_idx=name_idx, label_starting_idx=label_starting_idx, sanity_check=True, transforms = train_transform, classes=None)
    val_dataset, _ = prepare_dataset(img_dir, val_path, meta_path, image_name_idx=name_idx, label_starting_idx=label_starting_idx, sanity_check=True, transforms = val_transform, classes=None)
    test_dataset, _ = prepare_dataset(img_dir, test_path, meta_path, image_name_idx=name_idx, label_starting_idx=label_starting_idx, sanity_check=True, transforms = val_transform, classes=None)

    # Define a function to get the dataset given indices
    def get_dataset(indices, transform=None):
        df_subset = dataset.df.iloc[indices].reset_index(drop=True) # Reset index here
        return MultilabelDataset(df_subset, image_name_idx=name_idx,
                                label_starting_idx=label_starting_idx,
                                transforms=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=False, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=False, sampler=test_sampler)

    num_classes = 28
    
    writer = SummaryWriter(log_dir)

    model.projector = nn.Identity().cuda(gpu)
    model = FineTunedModel(model).cuda(gpu)
#convert to ddp
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if args.rank == 0:
        print('finetuning model loaded')

    # Optimizer and loss function settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-2)
    #hard coded number of epochs but should change to args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, 0.00001)


    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
    for epoch in range(40):
        model.train()
        running_loss = 0.0
        epoch_labels = []
        epoch_outputs = []
        print('Currently in at epoch {}'.format(epoch))
        train_sampler.set_epoch(epoch)
        print('Training Phase')
        pbar = tqdm(total=len(train_loader))
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data and labels to the GPU if it's available
            data, labels = data.cuda(gpu), labels.cuda(gpu)

            outputs = model(data)
            loss = criterion(outputs, labels.squeeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Apply sigmoid activation to output and keep track of it for metrics calculation
            sigmoid_outputs = torch.sigmoid(outputs)
            epoch_outputs.append(sigmoid_outputs.detach().cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())

            # Track loss and write it to TensorBoard
            running_loss += loss.item() * data.size(0)
            if args.rank == 0:
                writer.add_scalar('Train/Loss_per_batch', loss.item(), epoch * len(train_loader) + batch_idx)

            pbar.update(1)
        scheduler.step()

        pbar.close()
        #if scheduler is not None:

        #    lr = scheduler.get_last_lr()[0]
        #   writer.add_scalar('Train/Learning rate', lr, epoch)
        #    scheduler.step()

        # Save model after each epoch
        name = f'{checkpoint_save_dir}/checkpoint_lvt_ddp_{epoch}.pth'
        torch.save(model.state_dict(), name)

        # Compute and log average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        writer.add_scalar('Train/Loss_per_epoch', epoch_loss, epoch)

        # Compute and log metrics for the entire epoch
        epoch_labels = np.concatenate(epoch_labels, axis=0).squeeze(1)
        epoch_outputs = np.concatenate(epoch_outputs, axis=0)
        compute_and_log_metrics('Train', epoch_labels, epoch_outputs, epoch, writer, threshold=0.5)
        print('Loss for current epoch is {}'.format(epoch_loss))

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_labels = []
        val_outputs = []

        print('Validation Phase')
        val_sampler.set_epoch(epoch)
        with torch.no_grad():
          pbar = tqdm(total=len(val_loader))
          for data, labels in val_loader:
              data, labels = data.cuda(gpu), labels.cuda(gpu)
              outputs = model(data)
              sigmoid_outputs = torch.sigmoid(outputs)
              val_outputs.append(sigmoid_outputs.detach().cpu().numpy())
              val_labels.append(labels.cpu().numpy())
              pbar.update(1)
          pbar.close()

        val_labels = np.concatenate(val_labels, axis=0).squeeze(1)
        val_outputs = np.concatenate(val_outputs, axis=0)
        compute_and_log_metrics('Val', val_labels, val_outputs, epoch, writer, threshold=0.5)

    save_name = 'lvt_ddp'
    torch.save(model.state_dict(), f'{checkpoint_save_dir}/{save_name}_final_model.pth')

    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    pbar = tqdm(total=len(test_loader))
    with torch.no_grad():  # Deactivate gradients for the following code
        for data, labels in test_loader:
            # Move data and labels to the device the model is on
            data, labels = data.cuda(gpu), labels.cuda(gpu)
            outputs = model(data)
            # Apply sigmoid function to get probabilities between 0 and 1
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            y_true.append(labels.cpu().detach().numpy())
            y_pred.append(probs)
            pbar.update(1)
    pbar.close()

    y_true = np.vstack(y_true).squeeze(1)
    y_pred = np.vstack(y_pred)

    # Binarize predictions
    y_pred_bin = np.where(y_pred > 0.5, 1, 0)

    print()
    print("Micro-average quality numbers")
    precision = precision_score(y_true, y_pred_bin, average='micro', zero_division=0.0)
    recall = recall_score(y_true, y_pred_bin, average='micro', zero_division=0.0)
    f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0.0)

    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    print()
    print("Macro-average quality numbers")
    precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0.0)
    recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0.0)
    f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0.0)

    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    ap = average_precision_score(y_true, y_pred, average=None)

    # Calculate ROC curves
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    n=0
    total=0
    for k, v in roc_auc.items():
        if not np.isnan(v):
          total += v
          n += 1
    average_auc = total / n

    total = 0
    map = 0
    for score in ap:
      if not np.isnan(score):
        total += 1
        map += score
    map = map / total

    print()
    print('General Metrics')
    accuracy = hamming_score(y_true, y_pred_bin)
    print('Average accuracy over all labels (hamming score/exact match ratio): {}'.format(accuracy))
    print('Average AUC score: {}'.format(average_auc))
    print('Mean Average Precision: {}'.format(map))

    print('')
    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-label data')
    plt.legend(loc="lower right", bbox_to_anchor=(1.5, 0))
    plt.show()
    plt.savefig('AUC.png')

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = build_vmamba()
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [1024] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, input):
        return self.backbone(input)

    def forward_old(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

if __name__ == '__main__':
    seed_value = 43
    rank = int(os.environ.get("RANK", 0))
    set_seed_ddp(seed_value, rank)
    main()
