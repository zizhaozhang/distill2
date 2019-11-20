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
from .utils import *

class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, pretrained_embedding=None):
        """Set the hyper-parameters and build the layers."""
        super(RNN, self).__init__()
        
        if pretrained_embedding is not None:
            embed_weight = pretrained_embedding 
            embed_size = embed_weight.shape[1]
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.embed.weight.data.copy_(torch.from_numpy(embed_weight))
            self.fix_embed = True
            print ('\t --> load&init the pretrained embedding matrix [{}]'.format(embed_weight.shape))
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.init_weights()
            self.fix_embed = False
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # captions (batch, vocab_size)
        embeddings = self.embed(captions)
        if self.fix_embed:
            input_embeddings = Variable(embeddings.data, requires_grad=False)
        else:
            input_embeddings = embeddings
        hiddens, _ = self.lstm(input_embeddings)
        last_hidden = select_last(hiddens, lengths.data)
        hiddens = mask_textfeat(hiddens, lengths.data)

        return hiddens, last_hidden
    
    # def sample(self, features, states=None):
    #     """Samples captions for given image features (Greedy search)."""
    #     sampled_ids = []
    #     inputs = features.unsqueeze(1)
    #     for i in range(20):                                      # maximum sampling length
    #         hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
    #         outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
    #         predicted = outputs.max(1)[1]
    #         sampled_ids.append(predicted)
    #         inputs = self.embed(predicted)
    #         inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
    #     sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
    #     return sampled_ids.squeeze()

class TextCNN(nn.Module):
    def __init__(self, embed_size, vocab_size, pretrained_embedding=False):
        """Set the hyper-parameters and build the layers."""
        super(TextCNN, self).__init__()
        if pretrained_embedding:
            import pickle
            embed_weight = pickle.load(open('dataset/init_coco_glove_embeddings.pickle','rb'))
            embed_size = embed_weight.shape[1]
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.embed.weight.data.copy_(torch.from_numpy(embed_weight))
            print ('load&init the pretrained embedding matrix [{}]'.format(embed_weight.shape))
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.init_weights()
        
        # 1: unigram 2: bigram 3: trigram
        self.conv1 = nn.Conv1d(embed_size, 128, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(embed_size, 256, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(embed_size, 256, kernel_size=3, padding=0)

    def forward(self, captions, lengths):
        # captions (batch, vocab_size)
        # Return :  
        #   hiddens (batch, hidden_size)
        #   att_weights (batch, seq_len)
        embeddings = self.embed(captions)
        fixed_embeddings = Variable(embeddings.data, requires_grad=False)
        fixed_embeddings = fixed_embeddings.permute(0,2,1) # (batch_size, embed_size, seq_len)
        seq_len = fixed_embeddings.size(2)

        
        unigram = F.tanh(self.conv1(fixed_embeddings))
        unigram = F.max_pool1d(unigram, kernel_size=unigram.size(2)).squeeze()
        bigram = F.tanh(self.conv2(fixed_embeddings))
        bigram = F.max_pool1d(bigram, kernel_size=bigram.size(2)).squeeze()
        trigram = F.tanh(self.conv3(fixed_embeddings))
        trigram = F.max_pool1d(trigram, kernel_size=trigram.size(2)).squeeze()
        out = torch.cat([unigram, bigram, trigram], dim=1)

        return None, out

class NonLocalblock(nn.Module):
    # Input
        # A: (B, feat, N)
        # B: (B, feat , M)
    # Output
        # O (B, feat, M)
    def __init__(self, input_Adim, input_Bdim, embed_dim):
        super(NonLocalblock, self).__init__()
        self.in_A = nn.Conv2d(input_Adim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.in_B = nn.Conv2d(input_Bdim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.in_Bh = nn.Conv2d(input_Bdim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.out_B = nn.Conv2d(embed_dim, input_Bdim, kernel_size=1, padding=0, bias=False)
        self.out_B_bn = nn.BatchNorm1d(input_Bdim)
        self.norm_factor = np.sqrt(embed_dim)

    def forward(self, input_A, input_B):
        A = self.in_A(input_A.unsqueeze(3)).squeeze(3)
        B = self.in_B(input_B.unsqueeze(3)).squeeze(3)
        B_h =  self.in_Bh(input_B.unsqueeze(3)).squeeze(3)

        batch_size, feat_n, A_n = A.size()
        h = torch.bmm(A.permute(0,2,1), B) #(B, N, M)
        h.div_(self.norm_factor) # normalization
        att = F.softmax(h, dim=2) # apply softmax for each row to obtain (B, N, M)

        out_B = torch.bmm(B_h, att.permute(0,2,1)) # (B, feat, N)
        out_B = self.out_B(out_B.unsqueeze(3)).squeeze(3) # (B, input_B_dim, N)
        out_B = self.out_B_bn(out_B)
        # out_B = out_B + input_B put it outside

        return out_B, att

class AttentionGate(nn.Module):
    # Input
        # A: (B, feat, N)
        # B: (B, feat, 1) a sentence embedding
    # Output
        # O (B, feat, M)
    def __init__(self, input_Adim, input_Bdim, embed_dim):
        super(AttentionGate, self).__init__()
        self.in_A = nn.Conv1d(input_Adim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.in_B = nn.Conv1d(input_Bdim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.att_conv = nn.Conv1d(embed_dim, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, input_A, input_B):

        batch_size, img_feat_size, conv_feat_num = input_A.size()
        A = self.in_A(input_A)
        B = self.in_B(input_B)
        B = B.expand(batch_size, B.size(1), conv_feat_num)

        h = F.tanh(A + B)
        att = self.att_conv(h).squeeze(1)

        att = F.sigmoid(att) # TODO, sigmoid works better right?

        context_feat = torch.bmm(input_A, att.unsqueeze(2)).squeeze() #(B, multifeat_size)
       
        return context_feat, att

class MultiModal(nn.Module):
    def __init__(self, n_classes, img_feat_size, args):
        super(MultiModal, self).__init__()
        self.deathRate = args.death_rate

        self.nonlocalblock_img = NonLocalblock(img_feat_size, img_feat_size, args.attfeat_size)
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
                    nn.Dropout(args.last_drop_rate),
                    nn.Linear(img_feat_size, n_classes)
                    )

        # TODO check which kinds of initialiation is better. Xaiver is not that good
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

    def forward(self, img_feat, text_feat_tuple, lengths):
        # img_feat (B, img_feat_size, H, W)
        # text_feat (B, seq_len, hidden_size), seq_len = max(lengths)
        # lengths seq_len for each batch sample

        text_feat, text_feat_last = text_feat_tuple
        batch_size, img_feat_size, h, w = img_feat.size()
        conv_feat_num = h*w
        
        # compute nonlocal bloc
        img_feat_vec = img_feat.view(batch_size, img_feat_size, conv_feat_num)
        att_feat_img, _ = self.nonlocalblock_img(img_feat_vec, img_feat_vec)
        att_feat_img = att_feat_img + img_feat_vec # skip connection
        transfer_text = self.transfer(torch.mean(img_feat_vec,2))

        transfer_loss = F.mse_loss(transfer_text, text_feat_last.detach())

        transfer_text = transfer_text.unsqueeze(2)
        att_feat_textimg = None
        for i in range(self.num_rn_module):
            tf = text_feat if self.sampleGate() else transfer_text
            tmp, _ = getattr(self, 'nonlocalblock_text'+str(i))(tf, self.drop(img_feat_vec)) # (B, conv_feat_num, seq_len) 
            att_feat_textimg = (att_feat_textimg + tmp.mean(2)) if att_feat_textimg is not None else tmp.mean(2)

        att_feat_img = torch.mean(att_feat_img, 2)

        # feat = torch.cat([att_feat, masked_att_feat_textimg, torch.mean(img_feat_vec, 2)] , 1) 
        feat = torch.cat([att_feat_img, att_feat_textimg] , 1) 
        # perform skip connection of image feature
        # use two classification module to classify 
        out = self.cls(feat) 

        return out, transfer_loss


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
        # only coco has pretrained embedding
        pretrained_embedding = None
        if args.dataset in  ['coco', 'cub', 'vgnome']:
            import pickle
            pretrained_embedding = pickle.load(open('dataset/init_{}_glove_embeddings.pickle'.format(args.dataset),'rb'))

        self.rnn = RNN(args.embed_size, args.hidden_size, vocab_size, args.num_layers, pretrained_embedding)
        # self.rnn = TextCNN(args.embed_size, vocab_size, args.pretrained_embedding)
        # init the multimodel
        self.cls = MultiModal(n_classes, img_feat_size, args)
        self.activ_func = get_activation(args.dataset)

        print ('-> init a TandemNet2 (pretrained {}: {}  use glove embedding: {})'.format(model_name, pretrained, pretrained_embedding is not None))
        # print ('\t use every time step: {}'.format(self.use_every_timestep))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.rnn.parameters()) + list(self.cls.parameters())

    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def enable_text(self):
        self.cls.no_text_in_test = False
        print ('=> set no_text_in_test to False')

    def select_text_feat(self, txt_feat, txt_att_weights):
        
        selected_feat = []
        sorted_weight, indices = torch.sort(txt_att_weights, 1, descending=True)
        # need faster implementation
        for i in range(txt_feat.size(0)):
            tf = txt_feat[i]
            ids = indices[i][:self.text_feat_num]
            selected_feat.append(tf[ids])
        
        selected_feat = torch.stack(selected_feat, dim=0)
        return selected_feat
    
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
        logit, transfer_loss  = self.cls(img_feat, (multi_txt_feat, avg_txt_feat), lengths)

        return self.activ_func(logit), transfer_loss

class MultiLabelRelationResNet(nn.Module):
    def __init__(self, args, n_classes=80,
                 pretrained=True, model_name='resnet101'):
        super(MultiLabelRelationResNet, self).__init__()
        # init a CNN
        self.n_classes = n_classes
        cnn = resnet.__dict__.get(model_name)(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(cnn.children())[:-2]) # delete the last fc layer
        if model_name == 'resnet18':
            img_feat_size  = 512
        else:
            img_feat_size = 2048
        self.nonlocalblock_img = NonLocalblock(img_feat_size, img_feat_size, args.attfeat_size)
        self.cls = nn.Linear(img_feat_size, n_classes)
        self.activ_func = get_activation(args.dataset)

        self.cls.apply(weights_init)
        print ('-> init a MultiLabelRelationResNet (pretrained: {})'.format(pretrained))
        print ('\t last activation func: {}'.format(self.activ_func))

    def initial_parameters(self):
        return list(self.cls.parameters())

    def pretrained_parameters(self):
        return list(self.cnn.parameters())

    def forward(self, images):
        # Inputs:
            # images (B, img_feat_size, H, W)
           
        img_feat = self.cnn(images) 

        batch_size, img_feat_size, h, w = img_feat.size()
        conv_feat_num = h*w
        img_feat_vec = img_feat.view(batch_size, img_feat_size, conv_feat_num)
        att_feat_img, _ = self.nonlocalblock_img(img_feat_vec, img_feat_vec)
        att_feat_img += img_feat_vec

        logit = self.cls(torch.mean(att_feat_img, 2)) 

        return self.activ_func(logit)
