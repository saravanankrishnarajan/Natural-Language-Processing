import nltk
from nltk.tokenize import word_tokenize
import re 
import pandas as pd
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import Corpus

def PresenceAbsenceVectorization(txt_list):
    Corpus_list = Corpus.MakeCorpus(txt_list)
    Corpus_dict = dict.fromkeys(Corpus_list,0)
    #print(Corpus_dict)
    list_dict =[]
    
    for sent in txt_list:
        words_token = word_tokenize(re.sub('[@.!&{}\'<>*"]','',sent))
        New_Corpus_dict = Corpus_dict.copy()
        for word in words_token:
            New_Corpus_dict[word]=1
        list_dict.append(list(New_Corpus_dict.values()))
        
    return Corpus_list,list_dict
    

def CountVectorization(txt_list):
    regex1 = r'\b[a-zA-Z0-9]{1,}\b'
    count_vect = CountVectorizer(max_df=1.0,min_df=0.05,token_pattern  = regex1)
    X_counts = count_vect.fit_transform(txt_list)
    X_names = count_vect.get_feature_names_out()
    count_vect_df = pd.DataFrame(X_counts.toarray(),columns=X_names)
    return X_names,X_counts.toarray()

def TFIDFVectorization(txt_list):
    regex1 = r'\b[a-zA-Z0-9]{1,}\b'
    tf_vect = TfidfVectorizer(max_df=1.0,min_df=0.05,lowercase=True, stop_words=None,token_pattern  = regex1)
    tf_matrix = tf_vect.fit_transform(txt_list)
    tf_names = tf_vect.get_feature_names_out()
    tf_df = pd.DataFrame(tf_matrix.toarray(),columns=tf_names)

    return tf_names,tf_matrix.toarray()

