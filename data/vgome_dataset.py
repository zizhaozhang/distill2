
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

# see http://visualgenome.org/data_analysis/statistics for reference
label_corpus = ['man', 'window', 'person', 'tree', 'building', 'shirt', 'wall',
                'woman', 'sign', 'sky', 'ground', 'light', 'grass', 'cloud',
                'pole', 'car', 'table', 'leaf', 'hand', 'leg', 'head', 'water',
                'hair', 'people', 'ear', 'eye', 'shoe', 'plate', 'flower', 'line',
                'wheel', 'door', 'glass', 'chair', 'letter', 'pant', 'fence', 'train',
                'floor', 'street', 'road', 'hat', 'shadow', 'snow', 'jacket', 'boy',
                'boat', 'rock', 'handle']

label_corpus = ['man', 'window', 'tree', 'building', 'shirt', 'wall',
                 'sign', 'sky', 'ground', 'light', 'grass', 'pole',
                 'car', 'table', 'water', 'hair', 'shoe', 'plate',
                 'flower', 'door', 'glass', 'chair', 'fence', 'train',
                 'floor', 'road', 'hat', 'snow', 'boat', 'rock']


class VGomeDataset(data.Dataset):
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
        self.cat2label = {}
        for i, k in enumerate(label_corpus):
            self.cat2label[k] = i

        self.num_cats = len(self.cat2label) 

        # vgnome has varied number of annotations [1, 20], average 5.73
        # we still choose five as the parameter. It can be adjusted later on
        self.num_ann_onebatch = 5
        self.ids = [a for a in range(len(self.ann))]

        print('\t {} train samples from {} set'.format(len(self.ids), which_set ))
        print('\t {} of categories'.format(self.num_cats))

    def get_onehot_labels(self, cats):
        
        cats = np.array(cats)
        onehot_cat = np.zeros(self.num_cats, np.float32)
        onehot_cat[cats] = 1
        return torch.from_numpy(onehot_cat) #,onehot_cat

    def __getitem__(self, index):
        """Returns one data pair (image caption and labels)."""
        vocab = self.vocab
  
        img_name = self.img_list[index]
        targets = self.ann[img_name]['caption']
        img_cats = self.ann[img_name]['label']
        img_labels = self.get_onehot_labels([self.cat2label[a] for a in img_cats]) # re

        image = Image.open(os.path.join(self.img_root, img_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        if len(targets) > self.num_ann_onebatch:
            # picked_idx = np.random.choice(len(targets), self.num_ann_onebatch, replace=False)
            # picked_idx.sort()
            picked_idx = np.arange(self.num_ann_onebatch)
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
            target = target.replace('.','').replace(', ', ' ')
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            # caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)

        return image, captions, img_labels

    def __len__(self):
        return len(self.ids) 