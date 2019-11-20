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
import math
import itertools
from model.utils import *
from model.tandemnet2 import RNN, NonLocalblock, MultiLabelRelationResNet



class MultiModal(nn.Module):
    def __init__(self, n_classes, img_feat_size, args):
        super(MultiModal, self).__init__()
        self.deathRate = args.death_rate

        self.num_rn_module = args.num_rn_module
        for i in range(self.num_rn_module):
        	setattr(self, 'nonlocalblock_text'+str(i), NonLocalblock(args.hidden_size, img_feat_size, args.attfeat_size))

        self.drop2d = nn.Dropout2d(args.textimg_drop_rate)

        # visual to txt transfer
        self.transfer = nn.Sequential(
            nn.Linear(img_feat_size, img_feat_size//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(img_feat_size//2, args.hidden_size),
            nn.Tanh()
        )

		#### classifier
        self.cls = nn.Sequential(
                    nn.Linear(img_feat_size*2, img_feat_size),
                    nn.BatchNorm1d(num_features=img_feat_size),
                    nn.ReLU(),
                    nn.Dropout(args.last_drop_rate),
                    nn.Linear(img_feat_size, n_classes)
                    )

        self.apply(weights_init)

        self.no_text_in_test = True

    def drop(self, img_feat_vec):
        batch_size, conv_feat_num, seq_len = img_feat_vec.size()
        h_w = int(math.sqrt(seq_len))

        out = self.drop2d(img_feat_vec.view(batch_size, conv_feat_num, h_w, h_w))
        return out.view(batch_size, conv_feat_num, seq_len)

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

    def _forward(self, img_feat, text_feat_tuple):
        # img_feat (B, img_feat_size, H, W)
        # text_feat (B, seq_len, hidden_size), seq_len = max(lengths)

        text_feat, text_feat_last = text_feat_tuple
        batch_size, img_feat_size, h, w = img_feat.size()
        conv_feat_num = h*w

        # compute nonlocal bloc
        img_feat_vec = img_feat.view(batch_size, img_feat_size, conv_feat_num)
        att_feat_img = torch.mean(img_feat_vec, 2)

        transfer_text = self.transfer(att_feat_img)

        transfer_loss = F.mse_loss(transfer_text, text_feat_last.detach(), reduce=False)

        transfer_text = transfer_text.unsqueeze(2)
        att_feat_textimg = None
        self.attention = []
        for i in range(self.num_rn_module):
            tf = text_feat if self.sampleGate() else transfer_text
            tmp, att_tmp = getattr(self, 'nonlocalblock_text'+str(i))(tf, self.drop(img_feat_vec)) # (B, conv_feat_num, seq_len)
            att_feat_textimg = (att_feat_textimg + tmp.mean(2)) if att_feat_textimg is not None else tmp.mean(2)
            self.attention.append(att_tmp)

        att_feat_textimg.div_(self.num_rn_module)

        feat = torch.cat([att_feat_img, att_feat_textimg], 1)
        out = self.cls(feat)

        return out, transfer_loss

    def _sampling(self, img_feat, text_feat_tuple, is_multi_class):
        ' To examine the semantic attention method only'
        # img_feat (B, img_feat_size, H, W)
        # text_feat (B, seq_len, hidden_size), seq_len = max(lengths)

        text_feat, text_feat_last = text_feat_tuple
        batch_size, img_feat_size, h, w = img_feat.size()
        conv_feat_num = h*w

        # compute nonlocal bloc
        img_feat_vec = img_feat.view(batch_size, img_feat_size, conv_feat_num)
        att_feat_img = torch.mean(img_feat_vec, 2)

        transfer_text = self.transfer(att_feat_img)

        transfer_loss = F.mse_loss(transfer_text, text_feat_last.detach())

        transfer_text = transfer_text.unsqueeze(2)
        att_feat_textimg = None
        self.attention = []
        self.each_feat_textimg = []
        for i in range(self.num_rn_module):
            tf = text_feat if self.sampleGate() else transfer_text
            tmp, att_tmp=getattr(self, 'nonlocalblock_text' + str(i))(tf, self.drop(img_feat_vec)) # (B, conv_feat_num, seq_len)
            tmp = tmp.mean(2)
            self.each_feat_textimg.append(tmp)
            att_feat_textimg = (att_feat_textimg + tmp) if att_feat_textimg is not None else tmp
            self.attention.append(att_tmp)

        att_feat_textimg.div_(self.num_rn_module)

        feat = torch.cat([att_feat_img, att_feat_textimg], 1)
        out = self.cls(feat)

        def activ(x):
            return F.sigmoid(x) if is_multi_class else F.softmax(x)

        # ranking multiclass likelihood
        k = self.num_rn_module if is_multi_class else 1
        # labels are ranked. It is important to get the corresponded category name in attention_utils
        logits = activ(out)
        num_valid_labels = (logits > 0.5).sum(1)
        labels = logits.topk(k=k, dim=1)[1] # B x self.num_qrn_module

        # forward multiple times fo each feat_textimg
        # TODO att_feat_img is filled with 0 paty attention here
        ranked_logits = []
        for i in range(self.num_rn_module):
            tmp = activ(self.cls(torch.cat([att_feat_img.fill_(0), self.each_feat_textimg[i]], 1)))
            rlogit = torch.gather(tmp, 1, labels)
            ranked_logits.append(rlogit.view(batch_size, 1, k))

        # Pay attention
        # ranked_logits is in B * num_attention * logits_of_each classs
        # each row is the top-k class response of an attention map
        ranked_logits = torch.cat(ranked_logits, dim=1)

        '''the class ownership assigment attention map'''
        ## find the highest response for each attention map
        self.attention_class_inds = []

        for i in range(ranked_logits.shape[0]):
            # for all batch data
            ranked_logit = ranked_logits[i].data.clone() #[3, 3]
            # import pdb; pdb.set_trace()
            nlabel = num_valid_labels[i].data[0]
            if nlabel < k:
                # zero-out logit of unpredicted labels
                ranked_logit[:, nlabel:] = 0
            combs = [a for a in range(k)]
            combs = [a for a in itertools.permutations(combs)]
            best_v, preds = 0, []
            for comb in combs:
                idx = torch.LongTensor(comb).cuda().view(len(comb),-1)
                vs = torch.gather(ranked_logit, 1, idx)
                v = vs.sum()
                if v > best_v: best_v, preds=v, comb

            self.attention_class_inds.append(preds)

        return out, transfer_loss

    def forward(self, img_feat, text_feat_tuple, is_multi_class, rank_attention):
        if self.training or (not self.training and not rank_attention):
            return self._forward(img_feat, text_feat_tuple)
        else:
            return self._sampling(img_feat, text_feat_tuple, is_multi_class)


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
        self.is_multi_class = args.dataset in ['coco', 'chestxray', 'vgnome']
        # init a RNN
        # only coco has pretrained embedding
        pretrained_embedding = None
        if args.dataset in['coco', 'vgnome']:
            import pickle
            pretrained_embedding = pickle.load(open('dataset/init_{}_glove_embeddings.pickle'.format(args.dataset),'rb'))
            self.pretrained_embedding = pretrained_embedding

        self.rnn = RNN(args.embed_size, args.hidden_size, vocab_size, args.num_layers, pretrained_embedding)
        # init the multimodel
        self.cls = MultiModal(n_classes, img_feat_size, args)
        self.activ_func = get_activation(args.dataset, args.cmd)
        self.rank_attention = args.save_attention

        print ('-> init a TandemNet2-v2 (pretrained {}: {}  use glove embedding: {})'.format(model_name, pretrained, pretrained_embedding is not None))
        # print ('\t use every time step: {}'.format(self.use_every_timestep))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.rnn.parameters()) + list(self.cls.parameters())

    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def enable_text(self):
        self.cls.no_text_in_test = False
        print ('=> set no_text_in_test to False')

    def get_attentions(self, as_numpy=True):
        # return a list of attention

        if as_numpy:
            att = [a.data.cpu().numpy() for a in self.cls.attention]
        else:
            att = self.cls.attention

        if hasattr(self.cls, 'attention_class_inds'):
            att = (att, self.cls.attention_class_inds)

        return att

    def forward(self, images, captions, lengths):
        # Inputs:
            # images (B, img_feat_size, H, W)
            # captions (B, vocab_size)
            # lengths: lengths for each batch element
        img_feat = self.cnn(images)
        _, last_txt_feat = self.rnn(captions, lengths)  # (batch, seq_len, hidden_size) (batch, hidden_size)
        batch_size = img_feat.size(0)
        aug_batch_size, hidden_size = last_txt_feat.size()
        multi_txt_feat  = torch.transpose(last_txt_feat.view(batch_size, aug_batch_size//batch_size, hidden_size), 2, 1)
        avg_txt_feat = multi_txt_feat.mean(2)

        logit, transfer_loss  = self.cls(img_feat, (multi_txt_feat, avg_txt_feat), is_multi_class=self.is_multi_class, rank_attention=self.rank_attention)

        return self.activ_func(logit), transfer_loss
