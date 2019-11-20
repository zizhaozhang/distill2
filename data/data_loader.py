import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from .dataset_utils import Vocabulary
import sys
 

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, texts, labels = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    if type(texts[0][0]) is list:
        # each image has multiple sentence, e.g. captions[0][1] batch 0 caption 1
        # flatten to captions in [caption*batch], caption as first batch as second
        captions = []
        for text in texts:
            captions.extend([torch.Tensor(t) for t in text])
    else:
        captions = texts

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]      
    lengths = torch.IntTensor(lengths)  

    return images, targets, lengths, labels.squeeze()


def get_loader(root, dataset_name, which_set, 
                batch_size, shuffle, num_workers,
                unary_mode=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    
    # init data loader
    img_scale = 1
    if dataset_name == 'chestxray':
        mean_std = ((0.517, 0.517, 0.517), (0.158, 0.158, 0.158))
    elif dataset_name == 'bcidr':
        mean_std = ((0.657, 0.467, 0.605), (0.0127, 0.021, 0.015))
        # mean_std = ((0, 0, 0), (1, 1, 1))
    else:
        mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if which_set == 'train':
        if dataset_name in ['bcidr']:
            transform = transforms.Compose([ 
                transforms.RandomRotation(degrees=5),
                transforms.Resize(size=(256*img_scale,256*img_scale)),
                transforms.RandomCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean_std[0], mean_std[1]) 
                             ])
        elif dataset_name in ['chestxray']:
            transform = transforms.Compose([ 
                transforms.RandomRotation(degrees=5),
                transforms.Resize(size=(256*img_scale,256*img_scale)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean_std[0], mean_std[1]) 
                             ])
        else:
            transform = transforms.Compose([ 
                transforms.Resize(size=(256*img_scale,256*img_scale)),
                transforms.RandomCrop(224*img_scale),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean_std[0], mean_std[1]) 
                            ])
    else:
        transform = transforms.Compose([ 
                transforms.Resize(size=(256*img_scale,256*img_scale)), # best option
                transforms.CenterCrop(224*img_scale),
                transforms.ToTensor(), 
                transforms.Normalize(mean_std[0], mean_std[1])
                                    ])
    
    if unary_mode:
        from .coco_dataset import CocoDatasetUnary as CocoDataset 
        from .chestxray_dataset import ChestXrayDataset 
        from .bcidr_dataset import  BCIDRDatasetUnary as BCIDRDataset
        from .vgome_dataset import VGomeDataset
    else:
        from .coco_dataset import CocoDataset 
        from .chestxray_dataset import ChestXrayDataset 
        from .bcidr_dataset import BCIDRDataset as  BCIDRDataset 
        from .vgome_dataset import VGomeDataset

    print('-> init {} dataset {} loader (unary mode = {})'.format(dataset_name, which_set, unary_mode))
    if dataset_name == 'coco':
        vocab = pickle.load(open('data/vocab_corpus/vocab_coco.pkl','rb'))
        if which_set == 'test': which_set = 'val'
        dataset = CocoDataset(root=root+'/coco',
                        which_set=which_set,
                        vocab=vocab,
                        transform=transform)
    elif dataset_name == 'bcidr':
        vocab = pickle.load(open('data/vocab_corpus/vocab_bcidr.pkl','rb'))
        dataset = BCIDRDataset(root=root+'/BCIDR',
                        which_set=which_set,
                        vocab=vocab,
                        transform=transform)
    elif dataset_name == 'chestxray':
        vocab = pickle.load(open('data/vocab_corpus/vocab_chestxray.pkl','rb'))
        dataset = ChestXrayDataset(root=root+'/chest_xray',
                        which_set=which_set,
                        vocab=vocab,
                        transform=transform)
    elif dataset_name == 'vgnome':
        vocab = pickle.load(open('data/vocab_corpus/vocab_vgnome.pkl','rb'))
        dataset = VGomeDataset(root=root+'/vgnome',
                        which_set=which_set,
                        vocab=vocab,
                        transform=transform) 
    else:
        raise ValueError('invalid dataset name {}'.format(dataset_name))
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    
    # will not extract hidden state at these chars
    # ignore_word_idx = [0] # 0 is the empty pad value
    # symbols = ['<end>','<start>','<unk>'] + [c for c in "-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "]
    # for a in symbols:
    #     if a in vocab.word2idx.keys():
    #         ignore_word_idx.append(vocab(a))

    
    return data_loader, len(dataset), dataset.vocab.__len__() 

