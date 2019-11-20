
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from .dataset_utils import Vocabulary
from pycocotools.coco import COCO

class CocoDatasetUnary(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, which_set, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco_cap = COCO(root+'/annotations/captions_{}2014.json'.format(which_set))
        self.coco_label = COCO(root+'/annotations/instances_{}2014.json'.format(which_set))
        self.root = self.root + '/{}2014'.format(which_set)
        ids = list(self.coco_cap.imgs.keys())
        
        ''' drop images without image labels in coco_label '''
        drop_img_list = []
        for i, img_id in enumerate(ids):
            anns = self.coco_label.imgToAnns[img_id]
            if len(anns) == 0:
                drop_img_list.append(img_id)
    
        self.num_ann = 5
        self.img_list = list(set(ids) - set(drop_img_list))
        self.ids = [a for a in range(len(self.img_list) * self.num_ann)]
        self.vocab = vocab
        self.transform = transform
        
        
        # transfer categories id to labels
        categories = [a['id'] for a in self.coco_label.dataset['categories']]
        labels = [a for a in range(0,len(categories))]
        self.cat_to_label = {a:int(b) for a, b in zip(categories,labels)}
        self.label_to_cat = {self.cat_to_label[a['id']]: a['name'] for a, b in zip(self.coco_label.dataset['categories'],labels)}
        self.num_cats = len(labels)
        
        print('\t {} samples ({} discharded)'.format(len(self.ids), len(drop_img_list)))
        print ('\t {} categories'.format(self.num_cats))

    def label_to_name(self, label):
        return self.label_to_cat[label]

    def get_img_labels(self, img_id):
        ann_ids = self.coco_label.getAnnIds(imgIds=img_id)
        targets = self.coco_label.loadAnns(ann_ids)
        cats = []
        # image img_id could have more than one mask annotations
        for t in targets:
            cats.append(self.cat_to_label[t['category_id']])

        cats = np.array(cats)
        onehot_cat = np.zeros(self.num_cats, np.float32)

        # if len(cats) != 0: # Careful! some image id does not have annotations
        onehot_cat[cats] = 1
        # else:
        #     print (img_id, 'has no label')
        return torch.from_numpy(onehot_cat) #,onehot_cat

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco_cap = self.coco_cap
        vocab = self.vocab

        img_idx = index // self.num_ann
        text_idx = index % self.num_ann

        img_id = self.img_list[img_idx]
        ann_ids = coco_cap.getAnnIds(imgIds=img_id)
        anns = coco_cap.loadAnns(ann_ids)
        targets = [ann['caption'] for ann in anns]
        caption = targets[min(text_idx, len(targets)-1)]
        
        img_labels = self.get_img_labels(img_id) # return multi-class image labels
        path = coco_cap.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
      
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        # caption.append(vocab('<start>')) # we no need to start 
        caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, img_labels

    def __len__(self):
        return len(self.ids)

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, which_set, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco_cap = COCO(root+'/annotations/captions_{}2014.json'.format(which_set))
        self.coco_label = COCO(root+'/annotations/instances_{}2014.json'.format(which_set))
        self.root = self.root + '/{}2014'.format(which_set)
        ids = list(self.coco_cap.imgs.keys())

        ''' drop images without image labels in coco_label '''
        drop_img_list = []
        for i, img_id in enumerate(ids):
            anns = self.coco_label.imgToAnns[img_id]
            if len(anns) == 0:
                drop_img_list.append(img_id)
        self.ids = list(set(ids) - set(drop_img_list))
        self.vocab = vocab
        self.transform = transform
        
        # transfer categories id to labels
        categories = [a['id'] for a in self.coco_label.dataset['categories']]
        labels = [a for a in range(0,len(categories))]
        self.cat_to_label = {a:int(b) for a, b in zip(categories,labels)}
        self.label_to_cat = {self.cat_to_label[a['id']]: a['name'] for a, b in zip(self.coco_label.dataset['categories'],labels)}
        self.num_cats = len(labels)
        
        print('\t {} samples ({} discharded)'.format(len(self.ids), len(drop_img_list)))
        print ('\t {} categories'.format(self.num_cats))

    def label_to_name(self, label):
        return self.label_to_cat[label]

    def get_img_labels(self, img_id):
        ann_ids = self.coco_label.getAnnIds(imgIds=img_id)
        targets = self.coco_label.loadAnns(ann_ids)

        cats = []
        # image img_id could have more than one mask annotations
        for t in targets:
            cats.append(self.cat_to_label[t['category_id']])

        cats = np.array(cats)
        onehot_cat = np.zeros(self.num_cats, np.float32)

        # if len(cats) != 0: # Careful! some image id does not have annotations
        onehot_cat[cats] = 1
        # else:
        #     print (img_id, 'has no label')
        return torch.from_numpy(onehot_cat) #,onehot_cat

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco_cap = self.coco_cap
        vocab = self.vocab
    
        img_id = self.ids[index]
        ann_ids = coco_cap.getAnnIds(imgIds=img_id)
        anns = coco_cap.loadAnns(ann_ids)
        targets = [ann['caption'] for ann in anns]
        if len(targets) != 5:
            minl = min(len(targets), 5)
            targets = targets[:minl] + targets[:5-minl]
        # assert len(targets) == 5, 'length of targets is not 5 ({})'.format(len(targets))
        img_labels = self.get_img_labels(img_id) # return multi-class image labels
        path = coco_cap.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # Convert caption (string) to word ids.
        captions = []
        # img_label_names = self.get_img_label_names(img_id)
        for target in targets: 
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            ## caption.append(vocab('<start>')) # we no need to start 
            caption = [vocab(token) for token in tokens]
            captions.append(caption)

        # caption.append(vocab('<end>'))
        # captions = torch.Tensor(captions)
        return image, captions, img_labels
    
    def get_img_label_names(self, img_id):
        ann_ids = self.coco_label.getAnnIds(imgIds=img_id)
        targets = self.coco_label.loadAnns(ann_ids)

        cats = []
        # image img_id could have more than one mask annotations
        for t in targets:
            cats.append(self.label_to_cat[self.cat_to_label[t['category_id']]])

        return set(cats)

    def __len__(self):
        return len(self.ids)