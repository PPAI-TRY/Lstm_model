from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config

from operator import itemgetter
from keras.models import load_model
import pandas as pd
import os
import pdb
########################################
############ Data Preperation ##########
########################################
question = pd.read_csv('data/question.csv')
question = question['words']

# =============================================================================
# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')
# train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
# train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
# train = train[['label','words_x','words_y']]
# train.columns = ['label','q1','q2']
# train.to_csv('data/train_join.csv')
#
# test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
# test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
# test = test[['words_x','words_y']]
# test.columns = ['q1','q2']
# test.to_csv('data/test_join.csv')
# =============================================================================

df = pd.read_csv('data/train_join.csv')
is_similar = list(df['label'])
sentences1 = list(df['q1'])
sentences2 = list(df['q2'])
del df


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

####################################
######## Word Embedding ############
####################################


# creating word embedding meta data for word embedding 
tokenizer, embedding_matrix = word_embed_meta_data(question, CONFIG.embedding_dim, CONFIG.max_nb_words, 'data/word_embed.txt')

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}

## creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2


##########################
######## Training ########
##########################

siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, 
					    CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio,CONFIG.max_nb_words)

best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')


########################
###### Testing #########
########################

model = load_model(best_model_path)

df_test = pd.read_csv('data/test_join.csv')
sentences_test1 = list(df_test['q1'])
sentences_test2 = list(df_test['q2'])


## creating sentence pairs
test_sentence_pairs = [(x1, x2) for x1, x2 in zip(sentences_test1, sentences_test2)]
del sentences_test1
del sentences_test2


#test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),
#					   ('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])


result = model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1,batch_size=1024)

# 提交结果
submit_dir = './submit/'
if not os.path.exists(submit_dir ):
            os.makedirs(submit_dir )


submit = pd.DataFrame()
submit['y_pre'] = list(result[:,0])
submit.to_csv(submit_dir+'leak_similarity_result.csv',index=False)




#preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
#results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
#results.sort(key=itemgetter(2), reverse=True)
#print (results)












