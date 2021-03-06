# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Reshape, Flatten,Lambda,  Bidirectional,Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras import backend as K

import pdb
# std imports
import time
import gc
import os

from inputHandler import create_train_dev_set
from data_util import get_tfidf_dict


class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm, 
                 rate_drop_dense, activation_function, validation_split_ratio,max_nb_words,filter_sizes,
                 num_filters):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = activation_function
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        self.max_nb_words = max_nb_words
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    def train_model(self,sentences_pair, is_similar, embedding_meta_data,tfidf_dict ,model_save_directory='./'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass the each from sentences_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """
        
        tokenizer, embedding_matrix,embedding_index = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix'],embedding_meta_data['embedding_index']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio,embedding_index,tfidf_dict)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        nb_words = min(self.max_nb_words,len(tokenizer.word_index)) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # Creating LSTM Encoder
        lstm_layer = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))


        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer(embedded_sequences_2)

        # Creating CNN Encoder layer for First Sentence
        cnn_1 = self.conv(embedded_sequences_1)
        
        # Creating CNN Encoder layer for Second Sentence
        cnn_2 = self.conv(embedded_sequences_1)
        
        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(self.number_dense_units, activation=self.activation_function)(leaks_input)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, cnn_1,cnn_2 ,leaks_dense])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['binary_crossentropy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=200, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path

    def conv(self,input_x):
        
        input_x = Reshape((self.max_sequence_length,self.embedding_dim,1))(input_x)
        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')

        maxpool_0 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')
        maxpool_1 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')
        maxpool_2 = MaxPool2D(pool_size=(self.max_sequence_length - self.filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')

        c_0 = conv_0(input_x)
        c_1 = conv_1(input_x)
        c_2 = conv_2(input_x)
        
        m_0 = maxpool_0(c_0)
        m_1 = maxpool_1(c_1)
        m_2 = maxpool_2(c_2)
        
        concatenated_tensor = concatenate([m_0, m_1, m_2],axis=1)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(self.rate_drop_dense)(flatten)
        
        return dropout

    def update_model(self,saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix

        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path
