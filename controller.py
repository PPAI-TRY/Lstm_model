from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from data_util import get_tfidf_score_and_save, get_tfidf_dict
from config import siamese_config

from operator import itemgetter
from keras.models import load_model
import pandas as pd
import os
import pdb
import matplotlib.pyplot as plt

########################################
############ Data Preperation ##########
########################################
question = pd.read_csv('data/question.csv')

#get_tfidf_score_and_save(question.words,'data/tfidf_word.txt')
#get_tfidf_score_and_save(question.chars,'data/tfidf_char.txt',word_flag=False)

# =============================================================================
# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')
# train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
# train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
# train= train[['label','words_x','words_y','chars_x','chars_y']]
# train.columns = ['label','q1_word','q2_word','q1_char','q2_char']
# train.to_csv('data/train_word_char.csv')
# 
# test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
# test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
# test = test[['words_x','words_y','chars_x','chars_y']]
# test.columns = ['q1_word','q2_word','q1_char','q2_char']
# test.to_csv('data/test_word_char.csv')
# =============================================================================

df = pd.read_csv('data/train_word_char.csv')
is_similar = list(df['label'])
sentences1 = list(df['q1_word'])
sentences2 = list(df['q2_word'])
char1 = list(df['q1_char'])
char2 = list(df['q2_char'])
del df



#word = question["words"]
#count={}
#for i in range(len(word)):
#    word_list= word[i].split()
#    for w in word_list:
#        if w not in count:
#            count[w]=1
#        else:
#            count[w]+=1
#
#count_value=[100 if c > 100 else c for c in count.values()]


####################################
######## Configuration #############
####################################
class Configuration(object):
    """Dump stuff here"""

CONFIG = Configuration()
CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']
CONFIG.max_nb_words = siamese_config['MAX_NB_WORDS']
CONFIG.filter_sizes = siamese_config['FILTER_SIZES']
CONFIG.num_filters = siamese_config['NUM_FILTERS']
####################################
######## Word Embedding ############
####################################


# creating word embedding meta data for word embedding 
tokenizer, embedding_matrix,embedding_index = word_embed_meta_data(question["words"], CONFIG.embedding_dim, CONFIG.max_nb_words, 'data/word_embed.txt')

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix,
    'embedding_index':embedding_index
}

## creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
char_pair = [(x1, x2) for x1, x2 in zip(char1, char2)]
del sentences1
del sentences2

## get tfidf dictionary
#tfidf_dict = get_tfidf_score_and_save(question.words,'data/tfidf_word.txt',word_flag=True)
tfidf_dict = get_tfidf_dict('data/tfidf_word.txt')


##########################
######## Training ########
##########################

siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, 
					    CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio,
                    CONFIG.max_nb_words,CONFIG.filter_sizes,CONFIG.num_filters)

best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, tfidf_dict, model_save_directory='./')


########################
###### Testing #########
########################

model = load_model(best_model_path)

df_test = pd.read_csv('data/test_word_char.csv')
sentences_test1 = list(df_test['q1_word'])
sentences_test2 = list(df_test['q2_word'])


## creating sentence pairs
test_sentence_pairs = [(x1, x2) for x1, x2 in zip(sentences_test1, sentences_test2)]
del sentences_test1
del sentences_test2


#test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),
#					   ('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]


test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,CONFIG.max_sequence_length, embedding_index, tfidf_dict)


result = model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1,batch_size=1024)

# 提交结果
submit_dir = './submit/'
if not os.path.exists(submit_dir ):
            os.makedirs(submit_dir )


submit = pd.DataFrame()
submit['y_pre'] = list(result[:,0])
submit.to_csv(submit_dir+'result0618_add_features.csv',index=False)






#preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
#results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
#results.sort(key=itemgetter(2), reverse=True)
#print (results)












