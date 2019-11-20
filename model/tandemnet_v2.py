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
from model.utils import *
from model.tandemnet2 import RNN
from functools import partial


class AttentionBlock(nn.Module):
    def __init__(self, img_feat_size, args):
        super(AttentionBlock, self).__init__()
        self.deathRate = args.death_rate
        self.in_img_conv = nn.Sequential(
                nn.Dropout(args.multi_drop_rate),
                nn.Conv2d(img_feat_size, args.multifeat_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(args.multifeat_size) if args.dataset in ['coco', 'vgnome'] else nn.Tanh()
        )

        self.in_text_conv = nn.Sequential(
                nn.Dropout(args.multi_drop_rate),
                nn.Conv1d(args.hidden_size, args.multifeat_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(args.multifeat_size) if args.dataset in ['coco', 'vgnome'] else nn.Tanh()
        )

        self.multi_qfc = nn.Linear(args.multifeat_size, args.attfeat_size, bias=False)
        self.multi_ifc = nn.Linear(args.multifeat_size, args.attfeat_size, bias=False)
        self.ifeat_proj_conv = nn.Conv2d(args.multifeat_size, args.attfeat_size, kernel_size=1, padding=0, bias=False)
        self.qfeat_proj_conv = nn.Conv2d(args.multifeat_size, args.attfeat_size, kernel_size=1, padding=0, bias=False)
        self.att_conv = nn.Conv2d(args.attfeat_size, 1, kernel_size=1, padding=0, bias=False)


        # TODO check which kinds of initialiation is better. Xaiver is not that good
        self.apply(weights_init)

        self.att_func = partial(F.softmax, dim=1)

    def forward(self, img_feat, text_feat):
        # img_feat (B, img_feat_size, H, W)
        # text_feat (B, hidden_size, seq_len)

        batch_size, img_feat_size, h, w = img_feat.size()
        conv_feat_num = h*w
        _, hidden_size, text_feat_num = text_feat.size()
        # firstly embed the input
        in_ifeat_map = self.in_img_conv(img_feat) # (B, multifeat_size, h, w)
        multifeat_size = in_ifeat_map.size(1)
        img_feat_vec = img_feat.view(batch_size, img_feat_size, -1)

        in_ifeat = in_ifeat_map.view(batch_size, multifeat_size, -1) # (B, multifeat_size, conv_feat_num)
        in_qfeat = self.in_text_conv(text_feat) # (B, multifeat_size, text_feat_num)

        # compute image attention
        globqfeat = torch.mean(in_qfeat, 2)
        qfeatatt = self.multi_qfc(globqfeat) # (B, attfeat_size)
        attfeat_size = qfeatatt.size(1)
        qfeatatt = qfeatatt.unsqueeze(2)
        qfeatatt = qfeatatt.expand(batch_size, attfeat_size, conv_feat_num) # (B, attfeat_size, conv_feat_num)
        ifeatproj = self.ifeat_proj_conv(in_ifeat_map) # (B, att, h, w)
        ifeatproj = ifeatproj.view(batch_size, attfeat_size, -1)
        addfeat_i = F.tanh(qfeatatt + ifeatproj) # (B, attfeat_size, conv_feat_num)

        # compute text attention
        globifeat = torch.mean(in_ifeat, 2)
        ifeatatt = self.multi_ifc(globifeat)
        ifeatatt = ifeatatt.unsqueeze(2)
        ifeatatt = ifeatatt.expand(batch_size, attfeat_size, text_feat_num) #(B, attfeat_size, text_feat_num)
        qfeatproj = in_qfeat.unsqueeze(3)
        qfeatproj = self.qfeat_proj_conv(qfeatproj)
        qfeatproj = qfeatproj.squeeze(3) #(B, attfeat_size, text_feat_num)
        addfeat_q = F.tanh(ifeatatt + qfeatproj)

        attfeat_join = torch.cat([addfeat_i, addfeat_q], dim=2) #(B, attfeat_size, conv_feat_num+text_feat_num)
        attfeat_join = attfeat_join.unsqueeze(3)
        att = self.att_conv(attfeat_join).squeeze(dim=3)
        att = att.squeeze(dim=1)
        att = self.att_func(att) # (B, conv_feat_num+text_feat_num)

        joint_feat = torch.cat([in_ifeat, in_qfeat], dim=2) #(B, multifeat_size, conv_feat_num+text_feat_num)
        att_feat = torch.bmm(joint_feat, att.unsqueeze(2)).squeeze() #(B, multifeat_size)

        return att_feat, att

class MultiModal(nn.Module):
    def __init__(self, n_classes, img_feat_size, args):
        super(MultiModal, self).__init__()
        self.deathRate = args.death_rate
        self.num_rn_module = args.num_rn_module

        for i in range(self.num_rn_module):
        	setattr(self, 'attention_block'+str(i), AttentionBlock(img_feat_size, args))

        # classifier
        if args.dataset in ['bcidr', 'chestxray']:
            self.cls = nn.Sequential(
                    nn.Dropout(args.last_drop_rate),
                    nn.Linear(args.multifeat_size, n_classes)
                    )
        else:
            self.cls = nn.Sequential(
                        nn.Linear(args.multifeat_size, args.multifeat_size//2),
                        nn.Dropout(args.last_drop_rate),
                        nn.Linear(args.multifeat_size//2, n_classes)
                        )

        self.out_img_conv = nn.Sequential(
                nn.Conv2d(img_feat_size, args.multifeat_size, kernel_size=1, padding=0, bias=False),
        )
        self.no_text_in_test = True
        self.multifeat_size = args.multifeat_size

        print('\t init {} attention module'.format(self.num_rn_module))

    def sampleGate(self):
        self.gate = True
        if self.training:
            if np.random.rand() < self.deathRate:
                self.gate = False
        else:
            if self.no_text_in_test:
                self.gate = False
            else:
                self.gate = True
        return self.gate

    def forward(self, img_feat, text_feat, lengths):

        batch_size = img_feat.size(0)

        self.attention = []
        att_feat_textimg = None
        for i in range(self.num_rn_module):
            if not self.sampleGate():
                text_feat.data.fill_(0)
            #     text_feat_tmp = text_feat.mul(0)
            # else:
            #     text_feat_tmp = text_feat
            # else:
            #     if not self.training:
            #         # only do in testing
            #         text_feat_tmp = text_feat.mul(1-self.deathRate)
            #     else:
            #         text_feat_tmp = text_feat
            tmp, att_tmp = getattr(self, 'attention_block'+str(i))(img_feat, text_feat) # (B, conv_feat_num, seq_len)
            att_feat_textimg = (att_feat_textimg + tmp) if att_feat_textimg is not None else tmp
            self.attention.append(att_tmp)
        att_feat_textimg.div_(self.num_rn_module)

        final_feat = att_feat_textimg + torch.mean(self.out_img_conv(img_feat).view(batch_size, self.multifeat_size, -1), 2)

        # perform skip connection of image feature
        # use two classification module to classify
        out = self.cls(final_feat)

        return out

class DistillModel(nn.Module):
    def __init__(self, args, vocab_size, n_classes=80,
                 pretrained=True, model_name='resnet101'):
        super(DistillModel, self).__init__()
        self.n_classes = n_classes
        # init a CNN
        cnn = resnet.__dict__.get(model_name)(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2]) # delete the last fc layer
        if model_name == 'resnet18':
            img_feat_size  = 512
        else:
            img_feat_size = 2048
        # init a RNN
        pretrained_embedding = None
        if args.dataset in  ['coco', 'vgnome']:
            import pickle
            pretrained_embedding = pickle.load(open('dataset/init_{}_glove_embeddings.pickle'.format(args.dataset),'rb'))

        self.rnn = RNN(args.embed_size, args.hidden_size, vocab_size, args.num_layers, pretrained_embedding=pretrained_embedding)
        # init the multimodel

        self.cls = MultiModal(n_classes, img_feat_size, args)
        self.activ_func = get_activation(args.dataset)

        print ('-> init a TandemNet-v2 (pretrained {}: {}  use glove embedding: {})'.format(model_name, pretrained, pretrained_embedding is not None))
        print ('\t fix cnn: {}'.format(args.fix_cnn))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.rnn.parameters()) + list(self.cls.parameters())

    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def enable_text(self):
        self.cls.no_text_in_test = False
        print ('=> set no_text_in_test to False')

    def set_deathrate(self, v):
        self.cls.death_rate = v
        print ('=> adjust death rate to {}'.format(v))

    def get_attentions(self, as_numpy=True):

        if as_numpy:
            att = [a.data.cpu().numpy() for a in self.cls.attention]
        else:
            att = self.cls.attention

        return att

    def forward(self, images, captions, lengths):
        # Inputs:
            # images (B, img_feat_size, H, W)
            # captions (B, vocab_size)
            # lengths: lengths for each batch element
        img_feat = self.cnn(images)
        _, last_txt_feat = self.rnn(captions, lengths)  # (batch, seq_len, hidden_size)
        batch_size = img_feat.size(0)
        _, hidden_size = last_txt_feat.size()
        multi_txt_feat  = torch.transpose(last_txt_feat.view(batch_size, -1, hidden_size), 2, 1)
        logit  = self.cls(img_feat, multi_txt_feat, lengths)

        return self.activ_func(logit)

class MultiLabelResNet(nn.Module):
    def __init__(self, args, n_classes=80,
                 pretrained=True, model_name='resnet101'):
        super(MultiLabelResNet, self).__init__()
        # init a CNN
        self.n_classes = n_classes
        cnn = resnet.__dict__.get(model_name)(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2]) # delete the last fc layer
        if model_name =='resnet18':
            img_feat_size  = 512
        else:
            img_feat_size = 2048
        #self.cls = nn.Linear(img_feat_size, n_classes)

        self.cls = nn.Sequential(
            nn.Dropout(args.last_drop_rate),
            nn.Linear(img_feat_size, n_classes)

        )

        self.activ_func = get_activation(args.dataset)
        self.cls.apply(weights_init)
        print ('-> init a MultiLabelResNet (pretrained: {})'.format(pretrained))
        print ('\t fix cnn ({})'.format(args.fix_cnn))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.cls.parameters())

    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def forward(self, images):
        # Inputs:
            # images (B, img_feat_size, H, W)

        img_feat = self.cnn(images)
        img_feat_vec = torch.mean(img_feat.view(img_feat.size(0), img_feat.size(1), -1), 2)

        logit  = self.cls(img_feat_vec)

        return self.activ_func(logit)
