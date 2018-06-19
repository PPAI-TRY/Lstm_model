# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:26:37 2018

@author: LonelyFeaster
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def split_string_as_list_by_ngram(input_string,ngram_value):
    #print("input_string0:",input_string)
    input_string=input_string.split()
    #print("input_string1:",input_string)
    length = len(input_string)
    result_string=[]
    for i in range(length):
        if i + ngram_value < length + 1:
            
            result_string.append(input_string[i:i+ngram_value])
    #print("ngram:",ngram_value,"result_string:",result_string)
    return result_string




def compute_blue_ngram(x1_list,x2_list):
    """
    compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict={}
    count_dict_clip={}
    #1. count for each token at predict sentence side.
    for token in x1_list:
        token_str =  ",".join([t for t in token])
        if token_str not in count_dict:
            count_dict[token_str]=1
        else:
            count_dict[token_str]=count_dict[token_str]+1
    count=np.sum([value for key,value in count_dict.items()])

    #2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        token_str =  ",".join([t for t in token])
        if token_str in count_dict:
            if token_str not in count_dict_clip:
                count_dict_clip[token_str]=1
            else:
                count_dict_clip[token_str]=count_dict_clip[token_str]+1

    #3. clip value to ceiling value for that token
    count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
    count_clip=np.sum([value for key,value in count_dict_clip.items()])
    result=float(count_clip)/(float(count)+0.00000001)
    return result


def get_length_diff(input_string_x1,input_string_x2):
    length1=float(len(input_string_x1))
    length2=float(len(input_string_x2))
    length_diff=(float(abs(length1-length2)))/((length1+length2)/2.0)
    return length_diff


def get_sentence_diff_overlap_pert(input_word_x1,input_word_x2):
    #0. get list from string
    input_list1 = input_word_x1.split()
    input_list2 = input_word_x2.split()
    length1=len(input_list1)
    length2=len(input_list2)

    num_same=0
    same_word_list=[]
    #1.compute percentage of same tokens
    for word1 in input_list1:
        for word2 in input_list2:
           if word1==word2:
               num_same=num_same+1
               same_word_list.append(word1)
               continue
    num_same_pert_min=float(num_same)/float(max(length1,length2))
    num_same_pert_max = float(num_same) / float(min(length1, length2))
    num_same_pert_avg = float(num_same) / (float(length1+length2)/2.0)

    #2.compute percentage of unique tokens in each string
    input_list1_unique=set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1=float(len(input_list1_unique))/float(length1)
    num_diff_x2= float(len(input_list2_unique)) / float(length2)

# =============================================================================
#     if index==0:#print debug message
#         print("input_string_x1:",input_word_x1)
#         print("input_string_x2:",input_word_x2)
#         print("same_word_list:",same_word_list)
#         print("input_list1_unique:",input_list1_unique)
#         print("input_list2_unique:",input_list2_unique)
#         print("num_same:",num_same,";length1:",length1,";length2:",length2,";num_same_pert_min:",num_same_pert_min,
#               ";num_same_pert_max:",num_same_pert_max,";num_same_pert_avg:",num_same_pert_avg,
#              ";num_diff_x1:",num_diff_x1,";num_diff_x2:",num_diff_x2)
# =============================================================================

    diff_overlap_list=[num_same_pert_min,num_same_pert_max, num_same_pert_avg,num_diff_x1, num_diff_x2]
    return diff_overlap_list

def edit(input_char_x1, input_char_x2):
    str1 = input_char_x1.split()
    str2 = input_char_x2.split()
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    #print("matrix:",matrix)
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def cos_distance_bag_tfidf(input_string_x1, input_string_x2,word_vec_dict, tfidf_dict,tfidf_flag=True):
    #print("input_string_x1:",input_string_x1)
    #1.1 get word vec for sentence 1
    sentence_vec1=get_sentence_vector(word_vec_dict,tfidf_dict, input_string_x1,tfidf_flag=tfidf_flag)
    #print("sentence_vec1:",sentence_vec1)
    #1.2 get word vec for sentence 2
    sentence_vec2 = get_sentence_vector(word_vec_dict, tfidf_dict, input_string_x2,tfidf_flag=tfidf_flag)
    #print("sentence_vec2:", sentence_vec2)
    #2 compute cos similiarity
    numerator=np.sum(np.multiply(sentence_vec1,sentence_vec2))
    denominator=np.sqrt(np.sum(np.power(sentence_vec1,2)))*np.sqrt(np.sum(np.power(sentence_vec2,2)))
    cos_distance=float(numerator)/float(denominator+0.000001)

    #print("cos_distance:",cos_distance)   d(i,j)=|X1-X2|+|Y1-Y2|.
    manhattan_distance=np.sum(np.abs(np.subtract(sentence_vec1,sentence_vec2)))
    #print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance=300.0
    manhattan_distance=np.log(manhattan_distance+0.000001)/5.0

    canberra_distance=np.sum(np.abs(sentence_vec1-sentence_vec2)/np.abs(sentence_vec1+sentence_vec2))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance=np.log(canberra_distance+0.000001)/5.0

    minkowski_distance=np.power(np.sum(np.power((sentence_vec1-sentence_vec2),3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance=np.log(minkowski_distance+0.000001)/5.0

    euclidean_distance=np.sqrt(np.sum(np.power((sentence_vec1-sentence_vec2),2)))
    if np.isnan(euclidean_distance): euclidean_distance =300.0
    euclidean_distance=np.log(euclidean_distance+0.000001)/5.0

#    EJ(A,B)=(A*B)/(||A||^2+||B||^2-A*B)    
    jaccard_distance=numerator/(np.sum(np.power(sentence_vec1,2))+np.sum(np.power(sentence_vec2,2))-numerator+0.000001)
#    if np.isnan(jaccard_distance): jaccard_distance =300.0

    return cos_distance,manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance,jaccard_distance

def get_sentence_vector(word_vec_dict,tfidf_dict,word_string,tfidf_flag=True):
    word_list = word_string.split()
    vec_sentence=0.0
    for word in word_list:
        #print("word:",word)
        word_vec=word_vec_dict.get(word,None)
        word_tfidf=tfidf_dict.get(word,None)
        #print("word_vec:",word_vec,";word_tfidf:",word_tfidf)
        if word_vec is None is None or word_tfidf is None:
            continue
        else:
            if tfidf_flag==True:
                vec_sentence+=word_vec*word_tfidf
            else:
                vec_sentence += word_vec * 1.0
    vec_sentence=vec_sentence/(np.sqrt(np.sum(np.power(vec_sentence,2)))+0.000001)
    return vec_sentence


def get_tfidf_score_and_save(corpus,target_file,word_flag=True):
    target_object = open(target_file, 'w')
#    TfidfVectorizer=None #TODO TODO TODO remove this.
#    print("You need to import TfidfVectorizer first, if you want to use tfidif function.")
    if word_flag:
        vectorizer = TfidfVectorizer(analyzer=lambda x:x.split(' '),min_df=3,use_idf=1,smooth_idf=1,sublinear_tf=1)
    else:
        vectorizer = TfidfVectorizer(analyzer=lambda x:x.split(' '))
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    dict_word_tfidf=dict(zip(vectorizer.get_feature_names(), idf))
    for k,v in dict_word_tfidf.items():
        target_object.write(k+"|||"+str(v)+"\n")
    target_object.close()


def get_tfidf_dict(tfidf_file_path):
    infp = open(tfidf_file_path, 'r')
    lines = infp.readlines()
    tfidf_dict={}
    for li in lines:
        word,num=li.split("|||")
        tfidf_dict[word]=float(num)
    return tfidf_dict
        

def data_mining_features(input_string_x1,input_string_x2,word2vec_dict,tfidf_dict,n_gram=8):
    """
    get data mining feature given two sentences as string.
    1)n-gram similiarity(blue score);
    2) get length of questions, difference of length
    3) how many words are same, how many words are unique
    4) question 1,2 start with how/why/when(为什么，怎么，如何，为何）
    5）edit distance
    6) cos similiarity using bag of words
    :param input_string_x1:
    :param input_string_x2:
    :return:
    """

    #1. get blue score vector
    feature_list=[]
    #get blue score with n-gram
    for i in range(n_gram):
        x1_list=split_string_as_list_by_ngram(input_string_x1,i+1)
        x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
        blue_score_i_1 = compute_blue_ngram(x1_list,x2_list)
        blue_score_i_2 = compute_blue_ngram(x2_list,x1_list)
        feature_list.append(blue_score_i_1)
        feature_list.append(blue_score_i_2)

    #2. get length of questions, difference of length
    feature_list.append(get_length_diff(input_string_x1,input_string_x2))

    #3. how many words are same, how many words are unique
    sentence_diff_overlap_features_list=get_sentence_diff_overlap_pert(input_string_x1,input_string_x2)
    feature_list.extend(sentence_diff_overlap_features_list)
    
    #4.edit distance
    edit_distance=float(edit(input_string_x1, input_string_x2))/30.0
    feature_list.append(edit_distance)

    #6.cos distance from sentence embedding
    distance_list_word2vec = cos_distance_bag_tfidf(input_string_x1, input_string_x2, word2vec_dict, tfidf_dict)
    #distance_list2 = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict,tfidf_flag=False)
    #sentence_diffence=np.abs(np.subtract(sentence_vec_1,sentence_vec_2))
    #sentence_multiply=np.multiply(sentence_vec_1,sentence_vec_2)
    
    feature_list.extend(distance_list_word2vec)
    #feature_list.extend(list(sentence_diffence))
    #feature_list.extend(list(sentence_multiply))
    return feature_list