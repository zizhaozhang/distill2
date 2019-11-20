import argparse
import json
import math
import os, sys
from os.path import exists, join, split
import time
import numpy as np
import shutil
import scipy.misc as misc
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F

from misc.utils import *
# import extractors
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from functools import partial
from data.dataset_utils import vector2txt, renormalize_img, get_cats
from misc.attention_utils import generate_attention_sequence

torch.cuda.manual_seed_all(0)


class Trainer:
    def __init__(self, network, train_loader, test_loader,
                args, devices):

        self.args = args
        self.lock_bn = True and(args.dataset in['coco', 'vgnome'])
        if not self.lock_bn:
            #TODO for medical image dataset batch_norm should not be locked
            print('WARNING: batch norm is not locked in dataset ', args.dataset)
        # self.ignore_char_idx = ignore_char_idx
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.GLOBAL_ITER = 0
        self.nc = network.n_classes
        self.dynamic_deathrate = args.dynamic_deathrate
        # set up optimizer

        if args.dataset in ['coco', 'chestxray', 'vgnome']:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        if args.dataset in ['coco', 'vgnome']:
            self.metric = partial(coco_f1_score, topk=args.f1_topk) if args.dataset == 'coco' else partial(coco_f1_score, topk=999)
        elif args.dataset in ['chestxray']:
            self.metric = mul_cls_auc
        elif args.dataset in ['bcidr']:
            self.metric = mul_cls_accuracy

        self.optimizer_init = torch.optim.Adam(network.initial_parameters(), lr=args.mm_lr)
        if not args.fix_cnn:
            self.optimizer_ft = torch.optim.Adam(network.pretrained_parameters(), lr=args.cnn_lr)

        self.save_path = '{}/{}'.format(args.checkpoint_path, args.name)
        # parallel model,
        # to save or reload model use self.model.module
        self.model = torch.nn.DataParallel(network, device_ids=devices)

        ''' attention '''
        self.attention_savepath = os.path.join('checkpoints/' + self.args.name,
                            'attention_visualization_{}_text{}'.format(('wo', 'w')[args.use_text_in_test], ('', '_unary')[args.loader_unary_mode]))
        if not os.path.isdir(self.attention_savepath) and args.save_attention:
            os.mkdir(self.attention_savepath)

        print ('=> init Trainer in device ({})'.format(devices))
        print ('\t optimizer_ft {} optimizer_init {}'.format(hasattr(self, 'optimizer_ft'),hasattr(self, 'optimizer_init')))
        print ('\t criterion {}'.format(self.criterion))

    def update(self, loss):
        self.optimizer_init.zero_grad()
        if hasattr(self, 'optimizer_ft'):
            self.optimizer_ft.zero_grad()

        loss.backward()

        if hasattr(self.model.module, 'rnn'):
            #f use distill model, which contains rnn and multimodal cls, we clip gradients
            torch.nn.utils.clip_grad_norm(self.model.module.rnn.parameters(), self.args.grad_clip)

        self.optimizer_init.step()
        if hasattr(self, 'optimizer_ft'):
            self.optimizer_ft.step()

    def get_loss_weight(self, labels):
        # compute loss weight
        if not hasattr(self, 'weight'):
            self.weight = torch.cuda.FloatTensor(2)
        else:
            self.weight.fill_(0)

        tot = labels.size(0) * self.nc

        self.weight[0] = labels.sum() / tot
        self.weight[1] = 1 - self.weight[0]

        return self.weight

    def train_epoch(self, epoch, eval_score=None, print_freq=50):
        model = self.model
        loader  = self.train_loader
        criterion = self.criterion
        args = self.args

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        scores  = AverageMeter()
        model.train()
        # lock the batch_norm of pretrained cnn
        if self.lock_bn:
            model.module.cnn.eval()

        end = time.time()
        for i, (input, captions, lengths, labels) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input_var = to_var(input)
            captions_var = to_var(captions)
            labels_var = to_var(labels)
            lengths_var = to_var(lengths)
            # compute output and loss
            transfer_loss = 0.0
            if self.args.no_mm:
                logit = model(input_var)
                loss = criterion(logit, labels_var)
            else:
                logit = model(input_var, captions_var, lengths_var)
                if type(logit) == tuple:
                    logit, transfer_loss = logit
                    transfer_loss = transfer_loss.mean()

                loss = criterion(logit, labels_var)

                ''' rescale the loss '''
                loss *= args.loss_mult
                loss += transfer_loss
                if type(transfer_loss) is not float:
                    transfer_loss = float(transfer_loss.data.cpu().numpy())
            # measure accuracy and record loss
            losses.update(float(loss.data), input.size(0))
            scores.update(eval_score(logit, labels), 1)

            self.update(loss)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}, {tfloss:.4f}) '
                        'Score {top1.val:.3f} ({top1.avg:.3f}) '.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, tfloss=transfer_loss, top1=scores))
                sys.stdout.flush()
                self.WRITER.add_scalar('train/all_loss', losses.val, self.GLOBAL_ITER)
                self.WRITER.add_scalar('train/score', scores.avg, self.GLOBAL_ITER)

            self.GLOBAL_ITER += 1

    def train(self):
        model = self.model
        args = self.args
        best_prec1 = 0
        start_epoch = 0
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if not args.no_history:
                    start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']

                model.module.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                self.GLOBAL_ITER = start_epoch*len(self.train_loader) # so the tehnsoboard visialization will be connected
                print('epoch {}, best_score {}'.format(start_epoch, best_prec1))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return
        self.WRITER = SummaryWriter(self.save_path)
        if args.lr_decay_at != '':
            decay_at_epoch = [int(a) for a in args.lr_decay_at.split(',')]
        else:
            decay_at_epoch = [a*args.lr_decay for a in range(1, args.epochs//args.lr_decay+1)]
        print ('-> decay learning rate at ', decay_at_epoch)
        for epoch in range(start_epoch, args.epochs):
            #if epoch % args.lr_decay == 0 or start_epoch == epoch:
            if start_epoch == epoch:
                mm_lr = args.mm_lr
                cnn_lr = args.cnn_lr
            if epoch in decay_at_epoch:
                if hasattr(self, 'optimizer_ft'):
                    cnn_lr = adjust_learning_rate(args.cnn_lr, self.optimizer_ft, epoch, decay_at_epoch.index(epoch)+1, decay_rate=args.lr_decay_rate) # epoch//args.lr_decay
                mm_lr = adjust_learning_rate(args.mm_lr, self.optimizer_init, epoch, decay_at_epoch.index(epoch)+1, decay_rate=args.lr_decay_rate)


            print('Epoch: [{0}]\tmm_lr {1:.06f} \tcnn_lr {2:.06f}'.format(epoch, mm_lr, cnn_lr))

            if self.dynamic_deathrate:
                self.model.module.set_deathrate(epoch / (args.epochs-1))

            # train for one epoch
            self.train_epoch(epoch, eval_score=self.metric)
            # evaluate on validation set
            prec1 = self.validate(epoch, eval_score=self.metric)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = '{}/checkpoint_latest.pth.tar'.format(self.save_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 1 == 0:
                history_path = '{}/checkpoint_{:03d}.pth.tar'.format(self.save_path, epoch+1)
                shutil.copyfile(checkpoint_path, history_path)
            self.WRITER.add_scalar('train/lr', mm_lr, self.GLOBAL_ITER)
            self.WRITER.export_scalars_to_json("{}/tensorboard_all_scalars.json".format(self.save_path))



    def test(self):
        args = self.args
        model = self.model
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if not args.no_history:
                    start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']

                model.module.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                self.GLOBAL_ITER = start_epoch*len(self.train_loader) # so the tehnsoboard visialization will be connected
                print('epoch {}, best_score {}'.format(start_epoch, best_prec1))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return

        # evaluate on validation set
        loader_op  = self.test_loader.dataset

        score_list = self.validate(start_epoch, eval_score=self.metric)

    def validate(self, epoch, eval_score=None, print_freq=10, no_text=False):

        args = self.args
        model = self.model
        loader  = self.test_loader
        criterion = self.criterion

        batch_time = AverageMeter()
        losses = AverageMeter()
        scores = AverageMeter()

        if args.use_text_in_test:
            model.module.enable_text()

        model.eval()

        end = time.time()
        all_labels = []
        all_logits = []
        for i, (input, captions, lengths, labels) in enumerate(loader):
            # measure data loading time
            input_var = to_var(input, volatile=True)
            captions_var = to_var(captions, volatile=True)
            labels_var = to_var(labels, volatile=True)
            lengths_var = to_var(lengths, volatile=False)

            # compute output
            if self.args.no_mm:
                logit = model(input_var)
            else:
                logit = model(input_var, captions_var, lengths_var)

            if type(logit) == tuple:
                logit, transfer_loss = logit
                transfer_loss = transfer_loss.mean()
            else:
                transfer_loss = 0
            loss = criterion(logit, labels_var)
            ''' rescale the loss '''
            loss *= args.loss_mult
            loss += transfer_loss
            losses.update(float(loss.data), input.size(0))
            scores.update(eval_score(logit, labels), 1)
            all_labels.append(labels.cpu())
            all_logits.append(logit.data.cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Score {score.val:.3f} ({score.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time, loss=losses,
                    score=scores))
                sys.stdout.flush()

            # save attention
            if self.args.save_attention:
                attentions = model.module.get_attentions()
                categories = get_cats(to_numpy(logit), to_numpy(labels), loader.dataset)
                texts = vector2txt(to_numpy(captions), to_numpy(lengths), loader.dataset, loader.batch_size)
                images = renormalize_img(to_numpy(torch.transpose(torch.transpose(input, 1, 2), 2, 3)), self.args.dataset)
                generate_attention_sequence(self.args.name, images,
                                            attentions, texts, categories,
                                            savedir=os.path.join(self.attention_savepath, 'iter'+str(i)))

        # re-calculate mAP for all test data
        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        scores.reset() # reset to forget previous accumulations
        f1O = eval_score(all_logits, all_labels, class_wise=False)
        f1C = eval_score(all_logits, all_labels, class_wise=True)

        scores.update(f1C, 1)   # use the f1-C metric for val score

        if hasattr(self,'WRITER'):
            self.WRITER.add_scalar('val/score', scores.avg, epoch)
            self.WRITER.add_scalar('val/loss', losses.avg, epoch)

        return f1C
