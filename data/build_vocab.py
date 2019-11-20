'''Due to the Python3 module issues, this file should be executed at the root folder'''

import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import glob, json
from data.dataset_utils import Vocabulary

def build_vocab_coco(jsonfile, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(jsonfile)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_chestxray(jsonfile, threshold):
    """Build a simple vocabulary wrapper."""

    data = json.load(open(jsonfile,'r'))
    print ('found {} annotation files'.format(len(data)))
    counter = Counter()
    # ids = coco.anns.keys()
    for i, (k, sample) in enumerate(data.items()):
        caption = sample['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_bladder(jsonfile, threshold=0):
    """Build a simple vocabulary wrapper."""

    json_list = json.load(open(jsonfile,'r'))
    print ('found {} annotation files'.format(len(json_list)))
    counter = Counter()
    # ids = coco.anns.keys()
    for i, anno in enumerate(json_list.values()):
        captions = anno['caption']
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(json_list)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # words += ['nuclear_feature', 'nuclear_crowding', 'polarity',
    #                             'mitosis', 'nucleoli', 'conclusion'] 
    
    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    # vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_vgnome(jsonfile, threshold):
    """Build a simple vocabulary wrapper."""

    json_list = json.load(open(jsonfile,'r'))
    print ('found {} annotation files'.format(len(json_list)))
    counter = Counter()
    # ids = coco.anns.keys()
    for i, anno in enumerate(json_list.values()):
        captions = anno['caption']
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(json_list)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
  
    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    # vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
        
    return vocab

def main(args):
    if 'coco' in args.vocab_path:
        vocab = build_vocab_coco(jsonfile=args.caption_file,
                        threshold=args.threshold)
    elif 'chestxray' in args.vocab_path:
        vocab = build_vocab_chestxray(jsonfile=args.caption_file,
                            threshold=args.threshold)
    elif 'bcidr' in args.vocab_path:
        vocab = build_vocab_bladder(jsonfile=args.caption_file,
                            threshold=0)
    elif 'vgnome' in args.vocab_path:
        vocab = build_vocab_vgnome(jsonfile=args.caption_file,
                            threshold=args.threshold)
    else:
        raise ValueError('can not find ' + args.vocab_path)
    vocab_path = args.vocab_path
    if vocab.__class__.__module__ == "__main__":
        from .dataset_utils import Vocabulary

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print (vocab.word2idx)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, 
                        default='/usr/share/mscoco/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)

# python build_vocab.py --caption_file /media/zizhaozhang/zz_small/dataset/chest_xray/train_labels.json --vocab_path ./data/vocab_corpus/vocab_chestxray.pkl
# python data/build_vocab.py --caption_file /media/zizhaozhang/DataArchiveZizhao/Bladder/merged/Report/train_annotation.json --vocab_path ./data/vocab_corpus/vocab_bladderreport.pkl
# python data/build_vocab.py --caption_file /media/zizhaozhang/zz_small/dataset/vgnome/train_labels.json --vocab_path ./data/vocab_corpus/vocab_vgnome.pkl