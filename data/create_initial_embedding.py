"""
- ASSUMES: that "preprocess_captions.py" already has been run.
- DOES: creates a word embedding matrix (embeddings_matrix) using GloVe vectors.
- Used from coco embedding init. zizhao
"""

import numpy as np
import pickle
import os
from dataset_utils import Vocabulary
from nltk.corpus import wordnet
import nltk

captions_dir = "./dataset"
glove_embed_dir = "./dataset"
word_vec_dim = 300

# load the vocabulary from disk:
sys.path.append('./')
pkl_path = os.path.join(os.getcwd(), 'data/vocab_corpus/vocab_vgnome.pkl')
vocab = pickle.load(open(pkl_path,'rb'))
vocab_size = len(vocab)
print ('vocab_size', vocab_size)
# read all words and their corresponding pretrained word vec from file:
pretrained_words = []
word_vectors = []
with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file:
    for line in file:
        # remove the new line char at the end:
        line = line.strip()

        # seperate the word from the word vector:
        line_elements = line.split(" ")
        word = line_elements[0]
        word_vector = line_elements[1:]

        # save:
        pretrained_words.append(word)
        word_vectors.append(word_vector)

# create an embedding matrix where row i is the pretrained word vector
# corresponding to word i in the vocabulary:
c = 0
missing_words = []
embeddings_matrix = np.zeros((vocab_size, word_vec_dim), np.float32)
special_tokens = ["<pad>", "<start>", "<unk>", "<end>"]
special_token_embeddings = {a:np.random.rand(embeddings_matrix.shape[1]) for a in special_tokens}
for vocab_index, word in enumerate(vocab.word2idx.keys()):
    # if vocab_index % 100 == 0:
        # print (vocab_index, word)
        # log(str(vocab_index))

    if word not in special_tokens and word in pretrained_words: # (the special tokens are initialized with zero vectors)
        word_embedd_index = pretrained_words.index(word)
        word_vector = word_vectors[word_embedd_index]
        # convert into a numpy array:
        word_vector = np.array(word_vector)
        # convert everything to floats:
        word_vector = word_vector.astype(float)
        # add to the matrix:
        embeddings_matrix[vocab_index, :] = word_vector

    elif '<' not in word:
        # some words in coco is mis-spelled or too special
        # we save a list of them
        print ('missing {} word {}'.format(c, word))
        c += 1
        missing_words.append(word)
    elif word in special_tokens:
        # for a specal
        print ('set embedding for {}'.format(word))
        embeddings_matrix[vocab_index, :] = special_token_embeddings[word]


# find the closest word in the vocabulary of Glove
closet_dict = {a: {'d':float('inf'),'word': '' } for a in missing_words}
for word in pretrained_words:
    if word not in vocab.word2idx.keys(): continue

    for target_word in missing_words:
        # distance measure to find the closest word
        d = nltk.edit_distance(word, target_word)
        if d < closet_dict[target_word]['d']:
            closet_dict[target_word]['word'] = word
            closet_dict[target_word]['d'] = d
# assign the embeddings to those missing words
for word, v in closet_dict.items():
    word_embedd_index = pretrained_words.index(v['word'])
    vocab_index = vocab.word2idx[word]
    word_vector = np.array(word_vector)
    word_vector = word_vector.astype(float)
    word_vector = word_vectors[word_embedd_index]
    embeddings_matrix[vocab_index, :] = word_vector

    print ('missing word {} {} -> {}'.format(vocab_index, word, v['word']))

# save the embeddings_matrix to disk:
pickle.dump(embeddings_matrix,
        open(os.path.join(glove_embed_dir, "init_vgnome_glove_embeddings.pickle"), "wb"))
