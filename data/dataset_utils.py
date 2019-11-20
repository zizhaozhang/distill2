
import numpy as np


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def softmax(x):
    # a = np.exp(x)
    # b = np.sum(np.exp(x), axis=1)
    # b = np.tile(b[:, np.newaxis], (1, a.size(1)))
    # import pdb; pdb.set_trace()
    # return a / b
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_cats(logits, labels, loader):
    cats = []
    # for each batch data
    for logit, label in zip(logits, labels):
        if label.shape[0] > 1:
            # label is one hot convert to name
            lab = np.where(label > 0)[0]
            # make prediction ranked it is required by attenton_utils to show semantic attention
            valid_pred = np.where(logit > 0.5)[0]
            pred = np.argsort(logit)[::-1][:valid_pred.shape[0]] 
        else:
            lab = label
            pred = np.argmax(softmax(logit), 1)
        cat = [loader.label_to_cat[a] for a in lab]
        pred = [loader.label_to_cat[a] for a in pred]
        
        cats.append({'Labels': cat, 'Predictions': pred})
    return cats

def renormalize_img(imgs, dataset_name):
    if dataset_name == 'chestxray':
        mean_std = ((0.517, 0.517, 0.517), (0.158, 0.158, 0.158))
    elif dataset_name == 'bcidr':
        mean_std = ((0.657, 0.467, 0.605), (0.0127, 0.021, 0.015))
    else:
        mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for c in range(3):
        imgs[:, :, :, c] = (imgs[:, :, :, c] * mean_std[1][c]) + mean_std[0][c]
    
    return imgs
        

def vector2txt(data, lengths, loader, batch_size):
    # Input: label[batch_size, feature_num, seq_len]
    # Output: text_list [[] .. ]
    feature_num = data.shape[0] // batch_size
    label = np.reshape(data, newshape=(batch_size, feature_num, -1))
    lengths = np.reshape(lengths, newshape=(batch_size, feature_num))

    label = label.astype(np.int32)
    text_list = []
    text_list_verbose = []
    if len(label.shape) == 2:
        label = np.expand_dims(label,axis=0)
    assert(len(label.shape) == 3)
    batch_feature_len = []
    # for all samples
    for k in range(label.shape[0]):
        text = []
        tmp_idx = []
        # for all feature
        for i in range(lengths[k].size):
            subtext = []
            # for each time step
            for s in range(lengths[k][i]):
                subtext += [loader.vocab.idx2word[label[k][i][s]]]

            text.append(' '.join(subtext))
            tmp_idx.append(len(subtext))

        # batch_feature_len.append(tmp_idx)
        text_list.append(text)

    return text_list #, batch_feature_len       