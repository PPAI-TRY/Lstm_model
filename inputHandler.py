from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#from gensim.models import Word2Vec
import numpy as np
import pickle
import gc
import pdb

from data_util import data_mining_features

#def train_word2vec(documents, embedding_dim):
#    """
#    train word2vector over traning documents
#    Args:
#        documents (list): list of document
#        min_count (int): min count of word in documents to consider for word vector creation
#        embedding_dim (int): outpu wordvector size
#    Returns:
#        word_vectors(dict): dict containing words and their respective vectors
#    """
#    model = Word2Vec(documents, min_count=1, size=embedding_dim)
#    word_vectors = model.wv
#    del model
#    return word_vectors


def create_embedding_matrix(tokenizer, embedding_dim, max_nb_words, word_embed_path):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        embedding_dim (int): dimention of word vector

    Returns:

    """
    word_index  = tokenizer.word_index
    embeddings_index = {}
    with open(word_embed_path,'r') as f:
        for i in f:
            values = i.split(' ')
            word = str(values[0])
            embedding = np.asarray(values[1:],dtype='float')
            embeddings_index[word] = embedding
    print('word embedding',len(embeddings_index))
    
    
    nb_words = min(max_nb_words,len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(str(word).upper())
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
    return word_embedding_matrix,embeddings_index


def word_embed_meta_data(question, embedding_dim, max_nb_words, word_embed_path):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document

    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(question)
    embedding_matrix,embedding_index = create_embedding_matrix(tokenizer, embedding_dim, max_nb_words, word_embed_path)
#    word_vector = train_word2vec(documents, embedding_dim)
#    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
#    del word_vector
#    gc.collect()
    return tokenizer, embedding_matrix,embedding_index



def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio,word2vec_dict,tfidf_dict):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [data_mining_features(x1,x2,word2vec_dict,tfidf_dict,n_gram=8)
            for x1, x2 in sentences_pair]
#    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
#             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length,word2vec_dict,tfidf_dict):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    
    leaks_test = [data_mining_features(x1,x2,word2vec_dict,tfidf_dict,n_gram=8)
            for x1, x2 in test_sentences_pair]
#    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
#                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test
