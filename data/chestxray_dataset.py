
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import json
from .dataset_utils import Vocabulary
from collections import OrderedDict
np.random.seed(1234)

# this 14 unique labels are obtained from ChestXRay14 paper.
# Herina is not available in openI
label_corpus = [
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
    ## 'hernia', 
    'normal',
]



class ChestXrayDataset(data.Dataset):
    def __init__(self, root, which_set, vocab, transform=None, balance_class=False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.img_root = os.path.join(root, 'Img')
        self.ann = json.load(open(os.path.join(root, '{}_labels.json'.format(which_set)),'r'))

        self.vocab = vocab
        self.transform = transform
        self.img_list = list(self.ann.keys())
        # transfer categories id to labels
        self.cat2label = {}
        self.label_to_cat = {}
        self.cat2imgs = OrderedDict()
        for i, k in enumerate(label_corpus):
            self.cat2label[k] = i
            self.label_to_cat[i] = k
            self.cat2imgs[k] = []


        drop_img_list = []
        for i, img_id in enumerate(self.img_list):
            cap = self.ann[img_id]['caption']
            if len(cap) <= 1:
                drop_img_list.append(img_id)
            else:
                for cat in self.ann[img_id]['label']:
                    self.cat2imgs[cat].append(img_id)

        self.img_list = list(set(self.img_list) - set(drop_img_list))

        # balance data
        if which_set == 'train' and balance_class:
            max_aug_times = 5
            num_per_cats = [len(a) for a in self.cat2imgs.values()]
            max_n = max(num_per_cats) // 2
            for k, imls in self.cat2imgs.items():
                times = min(max_aug_times, max_n // len(imls))
                self.img_list += imls * times
                print ('\t {} x {} times'.format(k, times))
        
        self.ids = [a for a in range(len(self.img_list))]
        
        self.num_cats = len(self.cat2label)

         # vgnome has varied number of annotations [1, 23], average 7.018
        self.num_ann_onebatch = 7

        print('\t {} train samples from {} set ({} discharded)'.format(len(self.ids), which_set, len(drop_img_list)))
        print('\t {} of categories'.format(self.num_cats))
        print('\t {} sentences in a data'.format(self.num_ann_onebatch))

    def get_onehot_labels(self, cats):
        
        cats = np.array(cats)
        onehot_cat = np.zeros(self.num_cats, np.float32)
        onehot_cat[cats] = 1
        return torch.from_numpy(onehot_cat) #,onehot_cat

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        
        img_name = self.img_list[index]
        caption = self.ann[img_name]['caption']
        img_cats = self.ann[img_name]['label']
        img_labels = self.get_onehot_labels([self.cat2label[a] for a in img_cats]) # re

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 1]
        targets = [t.replace(',','') for t in targets if not t.isdigit()]
        # targets = sorted(targets, key=lambda x: len(x))[::-1]

        if len(targets) > self.num_ann_onebatch:
            picked_idx = np.random.choice(len(targets), self.num_ann_onebatch, replace=False)
            # picked_idx = np.arange(self.num_ann_onebatch)
            picked_idx.sort() # sort since the sentences are ordered 
            targets = [targets[i] for i in picked_idx]
        elif len(targets) < self.num_ann_onebatch:
            how_many_left = self.num_ann_onebatch - len(targets)
            picked_idx = []
            while how_many_left > 0:
                k = min(how_many_left, len(targets))
                picked_idx += [a for a in range(k)]
                how_many_left -= k
            # picked_idx = np.random.choice(len(targets), how_many_left, replace=how_many_left>len(targets))
            # picked_idx.sort()
            targets = targets + [targets[i] for i in picked_idx]
        assert(len(targets) == self.num_ann_onebatch)

        # Convert caption (string) to word ids.
        captions = []
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)
        return image, captions, img_labels

    def __len__(self):
        return len(self.ids)
        


class ChestXrayDatasetUnary(ChestXrayDataset):
    def __init__(self, root, which_set, vocab, transform=None, balance_class=False):
        super().__init__(root, which_set, vocab, transform, balance_class)

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        
        img_name = self.img_list[index]
        caption = self.ann[img_name]['caption']
        img_cats = self.ann[img_name]['label']
        img_labels = self.get_onehot_labels([self.cat2label[a] for a in img_cats]) # re

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 1]
        targets = [t.replace(',','') for t in targets if not t.isdigit()]
        # targets = sorted(targets, key=lambda x: len(x))[::-1]

        target = ' '.join(targets)

        tokens = nltk.tokenize.word_tokenize(str(target).lower())
        caption = [vocab(token) for token in tokens]
        caption = torch.Tensor(caption)

        return image, caption, img_labels