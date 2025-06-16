# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ===== Standard Library =====
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
from itertools import cycle

# ===== Third-Party Packages =====
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from matplotlib import pyplot as plt
from tqdm import tqdm

# ===== Torch & Torchvision =====
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as trans  # Only if you use 'trans' separately

# ===== timm & fvcore =====
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma
from timm.scheduler import CosineLRScheduler
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

# ===== Custom Project Modules =====
from config import get_config
from models import build_model, build_vmamba, build_vmamba_512, build_vmamba_new
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import (
    NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor,
    load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
)

from ASL import AsymmetricLoss

# ===== Sklearn Metrics =====
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss, jaccard_score,
    roc_auc_score, average_precision_score, auc, roc_curve
)

# ===== Optional: wandb =====
# import wandb

# --- argparse ---
parser = argparse.ArgumentParser(description='Barlow Twins Training for Finetuning on ADPv2 Dataset')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--checkpoint-save-dir', default='./checkpoint_save/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

#   ── Data File Paths ───
parser.add_argument('--img-dir', type=str, required=True, help='Root directory with images')
parser.add_argument('--meta-path', type=str, required=True, help='Path to metadata .csv')
parser.add_argument('--train-csv', type=str, required=True, help='Path to training csv annotation')
parser.add_argument('--val-csv', type=str, required=True, help='Path to validation csv annotation')
parser.add_argument('--test-csv', type=str, required=True, help='Path to test csv annotation')

# ─── Training Hyperparameters ───
parser.add_argument('--finetune-lr', type=float, default=1e-4,
                    help='Learning rate for the fine-tuning optimizer')
parser.add_argument('--finetune-weight-decay', type=float, default=5e-2,
                    help='Weight decay for the fine-tuning optimizer')
parser.add_argument('--cosine-t0', type=int, default=10,
                    help='T_0 for CosineAnnealingWarmRestarts scheduler')
parser.add_argument('--cosine-eta-min', type=float, default=1e-5,
                    help='Eta_min for CosineAnnealingWarmRestarts scheduler')
parser.add_argument('--gamma-neg', type=float, default=2.0,
                    help='gamma_neg for AsymmetricLoss')
parser.add_argument('--gamma-pos', type=float, default=1.0,
                    help='gamma_pos for AsymmetricLoss')
parser.add_argument('--loss-clip', type=float, default=0.0,
                    help='clip value for AsymmetricLoss')
parser.add_argument('--input-size', type=int, default=1024,
                    help='Height/width for Resize transform')
parser.add_argument('--hflip-prob', type=float, default=0.5,
                    help='Probability of horizontal flip')
parser.add_argument('--vflip-prob', type=float, default=0.5,
                    help='Probability of vertical flip')
parser.add_argument('--rotation-degrees', type=float, default=180,
                    help='Max degrees for RandomRotation')
parser.add_argument('--jitter-prob', type=float, default=0.2,
                    help='Probability to apply ColorJitter')
parser.add_argument('--jitter-brightness', type=float, default=0.2,
                    help='ColorJitter brightness')
parser.add_argument('--jitter-contrast', type=float, default=0.2,
                    help='ColorJitter contrast')
parser.add_argument('--jitter-saturation', type=float, default=0.1,
                    help='ColorJitter saturation')
parser.add_argument('--jitter-hue', type=float, default=0.05,
                    help='ColorJitter hue')
parser.add_argument('--norm-mean', type=float, nargs=3,
                    default=[0.8627, 0.6328, 0.7579],
                    help='Mean for Normalize')
parser.add_argument('--norm-std',  type=float, nargs=3,
                    default=[0.1758, 0.2738, 0.1852],
                    help='Std for Normalize')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold for metric binarization')
parser.add_argument('--checkpoint-prefix', type=str,
                    default='checkpoint_lvt_ddp',
                    help='Prefix for per-epoch checkpoint files')
parser.add_argument('--final-model-name', type=str,
                    default='vmamba_ddp',
                    help='Filename for final saved model')


# Helper Functions
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
        self.threshold = 0.8  # Default threshold for classification
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
        scoring = np.where(scores>=self.threshold, 1, 0)
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
                scores[i, ind] = 1 if tmp[i, ind] >=self.threshold else 0 ### Thersholder!!!
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

class MultilabelDataset(torch.utils.data.Dataset):

        def __init__(self, df, image_name_idx=0, label_starting_idx = 1, transforms = None):
            self.df = df
            if transforms:
              self.transforms = transforms
            else:
              self.transforms = trans.Compose([trans.Resize((720, 720)),
                trans.ToTensor()
                ])

            self.name_idx = image_name_idx
            self.start_idx = label_starting_idx

        def __len__(self):
            return len(self.df)

        def __getitem__(self,idx):
            img = Image.open(self.df.iloc[idx, self.name_idx]).convert("RGB")
            label = self.df.iloc[idx, self.start_idx:]

            return self.transforms(img), torch.Tensor([label])

def prepare_dataset(img_dir_path, csv_path, meta_path, image_name_idx=0, label_starting_idx=2,
                    sanity_check=False, transforms=None, classes = None):
  train_df = pd.read_csv(csv_path)
  train_df["patch_image_id"] = train_df["patch_image_id"].astype(str)

  print('len of df:', len(train_df), "from path:" , csv_path)

  train_df["patch_image_id"] = train_df["patch_image_id"].map(lambda x: os.path.join(img_dir_path, x + '_resized.png'))

  if classes is not None:
    train_df = train_df[['patch_image_id']+ classes]

  dataset = MultilabelDataset(train_df, image_name_idx, label_starting_idx, transforms=transforms)
  return dataset, train_df


class FineTunedModel(nn.Module):
    def __init__(self, backbone, num_classes=16):
        super().__init__()
        self.backbone = backbone
        # Assuming the output of the backbone is 1024-dimensional as per the last layer before the projector
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        # Pass the output through the new fully connected layer
        return self.fc(x)

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

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv), file=sys.stdout)
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # ─── Hyperparameters from args ───
    finetune_lr        = args.finetune_lr
    finetune_wd        = args.finetune_weight_decay
    T_0                = args.cosine_t0
    eta_min            = args.cosine_eta_min
    gamma_neg          = args.gamma_neg
    gamma_pos          = args.gamma_pos
    loss_clip          = args.loss_clip
    input_size         = args.input_size
    hflip_p            = args.hflip_prob
    vflip_p            = args.vflip_prob
    rot_deg            = args.rotation_degrees
    jitter_p           = args.jitter_prob
    jb, jc, js, jh     = (args.jitter_brightness,
                          args.jitter_contrast,
                          args.jitter_saturation,
                          args.jitter_hue)
    norm_mean, norm_std = args.norm_mean, args.norm_std
    threshold          = args.threshold
    ckpt_prefix        = args.checkpoint_prefix
    final_name         = args.final_model_name

    # ─── Data Transforms ───
    train_transform = trans.Compose([
        trans.Resize((input_size, input_size)),
        trans.RandomHorizontalFlip(p=hflip_p),
        trans.RandomVerticalFlip(p=vflip_p),
        trans.RandomRotation(degrees=rot_deg),
        trans.RandomApply([trans.ColorJitter(
            brightness=jb, contrast=jc, saturation=js, hue=jh
        )], p=jitter_p),
        trans.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
        trans.RandomErasing()
    ])
    val_transform = trans.Compose([
        trans.Resize((input_size, input_size)),
        trans.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    # ─── Prepare datasets ───
    train_dataset, _ = prepare_dataset(
        args.img_dir, args.train_csv, args.meta_path,
        image_name_idx=0, label_starting_idx=1,
        sanity_check=True, transforms=train_transform)
    val_dataset, _ = prepare_dataset(
        args.img_dir, args.val_csv, args.meta_path,
        image_name_idx=0, label_starting_idx=1,
        sanity_check=True, transforms=val_transform)
    test_dataset, _ = prepare_dataset(
        args.img_dir, args.test_csv, args.meta_path,
        image_name_idx=0, label_starting_idx=1,
        sanity_check=True, transforms=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    per_device_bs = args.batch_size // args.world_size

    train_loader = DataLoader(train_dataset, batch_size=per_device_bs,
                              num_workers=args.workers, pin_memory=False,
                              sampler=train_sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=per_device_bs,
                              num_workers=args.workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=per_device_bs,
                              num_workers=args.workers, pin_memory=False)

    writer = SummaryWriter(args.checkpoint_save_dir)

    # ─── Model, Optimizer, Scheduler ───
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True)

    # replace projector + add finetune head
    model.projector = nn.Identity().cuda(gpu)
    model = FineTunedModel(model).cuda(gpu)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=finetune_wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, eta_min=eta_min
    )

    # ─── Resume or start at zero ───
    ckpt_path = args.checkpoint_dir / 'checkpoint.pth'
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Resuming fine-tune from epoch {start_epoch}')
    else:
        start_epoch = 0

    # ─── Loss Function: Assymetric loss gives best results in unbalanced dataset such as ADPv2. Cross Entropy loss can also be tried ───
    criterion = AsymmetricLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=loss_clip
    )

    # ─── Training Loop ───
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        epoch_labels, epoch_outputs = [], []

        train_sampler.set_epoch(epoch)
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(gpu), labels.cuda(gpu)
            outputs = model(data)
            loss = criterion(outputs, labels.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sigmoid_out = torch.sigmoid(outputs).detach().cpu().numpy()
            epoch_outputs.append(sigmoid_out)
            epoch_labels.append(labels.cpu().numpy())

            running_loss += loss.item() * data.size(0)
            if args.rank == 0:
                writer.add_scalar(
                    'Train/Loss_per_batch', loss.item(),
                    epoch * len(train_loader) + batch_idx
                )

            pbar.update(1)

        scheduler.step(epoch + batch_idx / len(train_loader))
        pbar.close()

        # ─── Saving checkpoint ..───
        ckpt_file = args.checkpoint_save_dir / f'{ckpt_prefix}_{epoch}.pth'
        torch.save(model.state_dict(), ckpt_file)

        # ─── Epoch metrics ───
        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Train/Loss_per_epoch', epoch_loss, epoch)

        all_labels  = np.concatenate(epoch_labels, axis=0).squeeze(1)
        all_outputs = np.concatenate(epoch_outputs, axis=0)
        compute_and_log_metrics(
            'Train', all_labels, all_outputs, epoch, writer,
            threshold=threshold
        )

        # ─── Validation ───
        model.eval()
        val_labels, val_outputs = [], []
        pbar = tqdm(total=len(val_loader), desc='Validation')
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.cuda(gpu), labels.cuda(gpu)
                out = torch.sigmoid(model(data)).cpu().numpy()
                val_outputs.append(out)
                val_labels.append(labels.cpu().numpy())
                pbar.update(1)
            pbar.close()

        val_labels  = np.concatenate(val_labels, axis=0).squeeze(1)
        val_outputs = np.concatenate(val_outputs, axis=0)
        compute_and_log_metrics(
            'Val', val_labels, val_outputs, epoch, writer,
            threshold=threshold
        )

    # ─── Save final model ───
    final_file = args.checkpoint_save_dir / f'{final_name}_final_model.pth'
    torch.save(model.state_dict(), final_file)


    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    pbar = tqdm(total=len(test_loader))
    with torch.no_grad():  # Deactivate gradients for the following code
        for data, labels in val_loader:
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
