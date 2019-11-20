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
from .utils import *
from.tandemnet2 import RNN
from functools import partial

class MultiModal(nn.Module):
    def __init__(self, n_classes, img_feat_size, args):
        super(MultiModal, self).__init__()
        self.deathRate = args.death_rate
        self.in_img_conv = nn.Sequential(
                nn.Dropout(args.multi_drop_rate),
                nn.Conv2d(img_feat_size, args.multifeat_size, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(args.multifeat_size) if args.dataset in ['coco', 'vgnome'] else nn.Tanh()
        )
        self.out_img_conv = nn.Sequential(
                nn.Conv2d(img_feat_size, args.multifeat_size, kernel_size=1, padding=0, bias=False),
                # nn.Tanh()
                # nn.BatchNorm2d(args.multifeat_size)
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

        # TODO check which kinds of initialiation is better. Xaiver is not that good
        self.apply(weights_init)
         
        self.no_text_in_test = True
        if args.dataset in ['coco', 'vgnome']:
            self.att_func = F.sigmoid
        else:
            self.att_func = partial(F.softmax, dim=1)
        print ('\t use attention function', self.att_func)


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

    def updateOutput(self, img_feat, text_feat, lengths):
        # img_feat (B, img_feat_size, H, W)
        # text_feat (B, hidden_size, seq_len)
        # lengths seq_len for each batch sample
        
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
        att = self.att_conv(attfeat_join).squeeze()
        att = self.att_func(att) # (B, conv_feat_num+text_feat_num)
        self.attention = [att]

        joint_feat = torch.cat([in_ifeat, in_qfeat], dim=2) #(B, multifeat_size, conv_feat_num+text_feat_num)
        att_feat = torch.bmm(joint_feat, att.unsqueeze(2)).squeeze() #(B, multifeat_size)

        final_feat = att_feat + torch.mean(self.out_img_conv(img_feat).view(batch_size, multifeat_size, -1), 2)
        # perform skip connection of image feature
        # use two classification module to classify   
        out = self.cls(final_feat) #+ self.cls2(torch.mean(img_feat_vec,2))

        return out

    def forward(self, img_feat, text_feat, lengths):
        
        if not self.sampleGate():
            text_feat.data.fill_(0)
        output = self.updateOutput(img_feat, text_feat, lengths)

        return output

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
        if args.dataset in  ['coco', 'cub', 'vgnome']:
            import pickle
            pretrained_embedding = pickle.load(open('dataset/init_{}_glove_embeddings.pickle'.format(args.dataset),'rb'))
        # self.rnn = RNNAtt(args.embed_size, args.hidden_size, vocab_size, args.num_layers)
        self.rnn = RNN(args.embed_size, args.hidden_size, vocab_size, args.num_layers, pretrained_embedding=pretrained_embedding)
        # init the multimodel

        self.cls = MultiModal(n_classes, img_feat_size, args)
        self.activ_func = get_activation(args.dataset)
        
        print ('-> init a TandemNet (pretrained {}: {}  use glove embedding: {})'.format(model_name, pretrained, pretrained_embedding is not None))
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
        self.rank_attention = args.save_attention
        self.activ_func = get_activation(args.dataset, args.cmd)

        # for CAM
        params = list(self.parameters())
        self.fc_W = Variable(params[-2].data, requires_grad=False).cuda()

        self.cls.apply(weights_init)
        print ('-> init a MultiLabelResNet (pretrained: {})'.format(pretrained))
        print ('\t fix cnn ({})'.format(args.fix_cnn))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.cls.parameters())
        
    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def get_attentions(self, as_numpy=True):
        # return a list of attention

        if as_numpy:
            att = [a.data.cpu().numpy() for a in self.attention]
        else:
            att = self.cls.attention

        if hasattr(self, 'attention_class_inds'):
            att = (att, self.attention_class_inds)

        return att

    def _forward(self, images):
        # Inputs:
            # images (B, img_feat_size, H, W)
           
        img_feat = self.cnn(images)   
        img_feat_vec = torch.mean(img_feat.view(img_feat.size(0), img_feat.size(1), -1), 2)

        logit  = self.cls(img_feat_vec)

        return self.activ_func(logit)

    def _sampling(self, images, k=3):
        img_feat = self.cnn(images)
        batch_size, feat_dim, h, w = img_feat.size()
        img_feat_vec = torch.mean(img_feat.view(img_feat.size(0), img_feat.size(1), -1), 2)

        logit  = self.cls(img_feat_vec)

        labels = logit.topk(k=k, dim=1)[1] # B x k
        num_valid_labels = (logit > 0.5).sum(1)

        # CAM method get columes of weights
        self.attention_class_inds = []
        tmp_attention = []
        
        for i in range(batch_size):
            conv_feat = img_feat[i].view(feat_dim, -1)
            W = torch.index_select(self.fc_W, 0, labels[i])
            att = torch.mm(W, conv_feat).view(W.size(0), -1)
            tmp_attention.append(att)
            self.attention_class_inds.append([0,1,2])

        tmp_attention = torch.stack(tmp_attention)
        tmp_attention = tmp_attention.transpose(0, 1)
        self.attention = [a for a in tmp_attention]

        return self.activ_func(logit)


    def forward(self, img_feat):
        if self.training or (not self.training and not self.rank_attention):
            return self._forward(img_feat)
        else:
            return self._sampling(img_feat)


