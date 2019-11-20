import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from misc.utils import weights_init
from torch.nn import functional as F
import torchvision.models.resnet as resnet
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np



def get_activation(dataset, mode='train'):
    if mode == 'test':
        return F.sigmoid if dataset in ['coco', 'chestxray', 'vgnome'] else lambda x: x
    else:
        return lambda x: x
    # 

def get_mask_attention_weight(att_weights, lengths):
    # set att_weight (B, max(lengths), N) 
    
    mask = Variable(torch.FloatTensor(att_weights.size()).fill_(0).cuda(), requires_grad=False)
    for i in range(mask.size(0)):
        mask[i, :lengths[i]] = 1
    # att_weights = att_weights * mask
    return att_weights

def mask_attention_weight(att_weights, lengths):
    # set att_weight (B, max(lengths), N) 
    
    mask = Variable(torch.FloatTensor(att_weights.size()).fill_(0).cuda(), requires_grad=False)
    for i in range(mask.size(0)):
        mask[i, :lengths[i]] = 1
    att_weights = att_weights * mask
    return att_weights


def mask_textfeat(text_feat, lengths):
    # text_feat (B, seq_len, hidden_size)

    mask = Variable(torch.FloatTensor(text_feat.size()).fill_(0).cuda(), requires_grad=False)
    for i in range(mask.size(0)):
        mask[i, :lengths[i],:] = 1
    text_feat = text_feat * mask
    return text_feat

def select_last(x, lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i]-1].fill_(1)
    # mask = Variable(mask, requires_grad=False)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x

