
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

class BCIDRDatasetUnary(data.Dataset):
    def __init__(self, root, which_set, vocab, transform=None):
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
        self.num_ann = 1
        self.num_cats = 4
        self.ids = [a for a in range(len(self.ann) * self.num_ann)]
        self.cat2label = {}

        print('\t {} samples from {} set'.format(len(self.ids), which_set ))

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        img_id = index // self.num_ann
        text_id = np.random.randint(0,4,1)[0]

        img_name = self.img_list[img_id]
        caption = self.ann[img_name]['caption'][text_id]
        img_labels = np.array([self.ann[img_name]['label']], np.int64)

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
      
        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 0] 
        assert(len(targets) == 5)
        
        # Convert caption (string) to word ids.
        captions = []
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)

        return image, captions, torch.from_numpy(img_labels)

    def __len__(self):
        return len(self.ids) 

label2word = {
    0: 'normal',
    1: 'low grade',
    2: 'high grade',
    3: 'insufficient information'
}

class BCIDRDataset(data.Dataset):
    def __init__(self, root, which_set, vocab, transform=None):
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
        self.num_ann = 5

        self.num_cats = 4
        self.ids = [a for a in range(len(self.ann) * self.num_ann)]
        self.cat2label = {}

        print('\t {} samples from {} set'.format(len(self.ids), which_set ))

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
        img_id = index // self.num_ann
        text_id = index % self.num_ann 

        img_name = self.img_list[img_id]
        caption = self.ann[img_name]['caption'][text_id]
        img_labels = np.array([self.ann[img_name]['label']], np.int64)

        image = Image.open(os.path.join(self.img_root, img_name+'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        targets = [c.lstrip(' ') for c in caption.split('.') if len(c) > 0] + [label2word[img_labels[0]]]
        assert(len(targets) == 6)
        
        # Convert caption (string) to word ids.
        captions = []
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)
            
        return image, captions, torch.from_numpy(img_labels)

    def __len__(self):
        return len(self.ids)