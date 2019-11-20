import argparse
import json
import math
import os
import pdb
from os.path import exists, join, split
import numpy as np
import time
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import multiprocessing as mp
import scipy.misc as misc
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_recall_fscore_support
from collections import OrderedDict


def mapk(logits, labels, topk=3, threshold=0.5, return_ap=False):
    ''' For final evaluation, need to put all test logits and labels together'''
    # logits [N, num_classes]
    # labels [N, num_classes]
    # topk: default (3) is used for coco
    def apk(actual, predicted, topk):
        score = 0.0
        num_hits = 0.0
        predicted = predicted[:min(len(predicted), topk)]
        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), topk)
    
    labels  = to_numpy(labels)
    logits = to_numpy(logits)
        
    # extract topk labels for each sample 
    # and guarantee the prob is over threshold
    predicteds = np.zeros(labels.shape)
    for i, logit in enumerate(logits):
        logit[logit<threshold] = 0
        predicted = np.argsort(logit)[::-1] # in decending order       
        predicted = [a for a in predicted[:min(len(predicted), topk)] if logit[a] > 0]  
        predicteds[i][predicted] = 1
        
    AP = []
    for i in range(labels.shape[1]):
        act = np.where(labels[:,i] == 1)[0]
        pre = np.where(predicteds[:,i] == 1)[0]
        AP.append(apk(act.tolist(), pre.tolist(), topk=100))

    if return_ap:
        return np.mean(AP), AP
    else:
        return np.mean(AP) # mAP

def convert_to_onehot_labels(labels, num_cls):
        
    onehot_cat = np.zeros((labels.shape[0], num_cls), np.float32)
    for i, label in enumerate(labels):
        onehot_cat[i, label] = 1
    return onehot_cat #,onehot_cat

def coco_f1_score(logits, labels, topk=3, threshold=0.5, class_wise=None):
    # logits [N, num_classes]
    # labels [N, num_classes] or [N]
    # topk: default (3) is used for coco
    # class_wise: if compute scores per class (default for class_wise=None). Otherwise, compute per sample
    
    labels  = to_numpy(labels)
    logits = to_numpy(logits)
    batch_size = logits.shape[0]
    eps = 0.0000000000000001
    if len(labels.shape) == 1 and batch_size != 1:
        labels = convert_to_onehot_labels(labels, num_cls=logits.shape[1])
        topk = 1
    # extract topk labels for each sample 
    # and guarantee the prob is over threshold
    predicteds = np.zeros(labels.shape)
    for i, logit in enumerate(logits):
        logit[logit<=threshold] = 0
        predicted = np.argsort(logit)[::-1] # in decending order       
        predicted = [a for a in predicted[:min(len(predicted), topk)] if logit[a] > 0]   
        predicteds[i][predicted] = 1

    if class_wise is not None and not class_wise:
        labels = np.reshape(labels,-1) 
        predicteds =  np.reshape(predicteds,-1) 
    
    tp = (labels * predicteds)
    num_tp = np.sum(tp, 0)+eps
    num_pred = np.sum(predicteds, 0)+eps
    num_p = np.sum(labels, 0)+eps
    precision = num_tp / (num_pred)
    recall = num_tp / num_p
    f1  = 2*precision*recall / (precision + recall)

    if class_wise is not None:
        print('Score (C: {0} TopK: {1}) \n \
            F1: {2:.3f} \t P {3:.3f} \t R {4:.3f}'
            .format(class_wise, topk, np.mean(f1), np.mean(precision), np.mean(recall)))

    return np.mean(f1)

def mul_cls_f1_score(logits, labels, threshold=0.5, class_wise=None):
    
    labels  = to_numpy(labels).astype(np.uint8)
    logits = to_numpy(logits)
    y_pred = (logits > threshold).astype(labels.dtype)

    f1, precision, recall, _ = precision_recall_fscore_support(labels, y_pred, average='micro')


    if class_wise is not None and class_wise:
         print('Score \n \
            F1: {0:.3f} \t P {1:.3f} \t R {2:.3f}'
            .format(f1, precision, recall))

    return f1


def mul_cls_auc(logits, labels, class_wise=None):
    # logits [N, num_classes]
    # labels [N, num_classes]
    # this is used for chestxray 
    chestxray_label_corpus = [
            'atelectasis',
            'cardiomegaly',
            'effusion',
            'infiltration',
            'mass',
            'nodule',
            'pneumonia',
            'pneumothorax',
            'consolidation',
            'edema',
            'emphysema',
            'fibrosis',
            'pleural_thickening', 
            'normal',
        ]

    labels  = to_numpy(labels).astype(np.uint8)
    logits = to_numpy(logits)

    AUCs = []
    for i in range(labels.shape[1]):
        if np.unique(labels[:,i]).size == 1:
            # when all labels are zero is not supported to compute in roc_auc_curve
            auc_cls = 0
        else:
            auc_cls = roc_auc_score(y_true=labels[:,i], y_score=logits[:,i], average=None)
            AUCs.append(auc_cls)

    auc_mean = np.mean(AUCs) if len(AUCs) > 0 else 0
        
    if class_wise is not None and class_wise:
        cat2label = OrderedDict()
        for i, k in enumerate(chestxray_label_corpus):
            cat2label[k] = AUCs[i]
        print('Mean AUC: {}'.format(auc_mean))
        for k, v in cat2label.items():
            print ('\t {}: {:0.3f}'.format(k, v))

    return auc_mean

def mul_cls_accuracy(logits, labels, class_wise=None):
    # logits [N, num_classes]
    # labels [N]

    labels  = to_numpy(labels)
    logits = to_numpy(logits)
    # batch_size = logits.shape[0]
    # eps = 0.0000000000000001

    if len(labels.shape) != 1:
        labels = np.argmax(labels, axis=1)

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    if class_wise is not None and class_wise:
        print('Score : \n \
            Accuracy: {0:.3f}'
            .format(acc))
        cf = confusion_matrix(labels, preds)
        print ('Confuion matrix: ') 
        print(cf)

    return acc


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def to_numpy(data):
    if type(data) == np.ndarray or type(data) == np.array:
        return data
    if type(data) is Variable:
        out =  data.data
    else:
        out = data
    # return out.cpu().numpy()
    return out.cpu().detach().numpy()

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

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def weights_init(m, method='xavier'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            if method == 'normal':
                m.weight.data.normal_(0.0, 0.02)
            elif method == 'xavier':
                torch.nn.init.xavier_uniform(m.weight)

    elif classname.find('BatchNorm') != -1: 
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
        
def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]

def adjust_learning_rate(init_lr, optimizer, epoch, step, decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay_rate ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
    return lr

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind])
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)
       
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    folder = os.path.split(filename)[0]
    if is_best:
        shutil.copyfile(filename, folder+'/model_best.pth.tar')
