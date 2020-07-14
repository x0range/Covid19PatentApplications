""" GridSearchCV optimized LatentDirichletAllocation script.
    Parameters for the procedure can be set at the beginning of the main() function.
    The script expects a pickle file with two objects:
            1. the corpus as dict of strings with document ID as key
            2. a pandas dataframe with additional information (including date) with document ID as index

    Note: Script uses various bits of code from several stackoverflow.com posts.

    """

""" Module imports"""
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import os
#import nltk 
import pdb
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gzip

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

""" Prepare stemmer. Attempt to load nltk wordnet oif not already downloaded on machine."""
stemmer = SnowballStemmer("english")
try:
    stemmer.stem(WordNetLemmatizer().lemmatize("test", pos='v'))
except:
    import nltk
    nltk.download('wordnet')

""" Perplexity score optimizable LDA class. Inherits from 
    sklearn.decomposition.LatentDirichletAllocation, overwrites only the score() method.""" 
class LDACustomScore(LatentDirichletAllocation):
    def score(self, X, y=None):
        """ Score method using negative perplexity instead of Likelihood as the standard
            score() method does. Negative because GridSearchCV maximizes and perplexity 
            should be minimized. 
            Arguments: 
                self: Instance
                X: array (Document word matrix)
                y: (ignored)
            Returns: float (negative perplexity score)"""
        perplexity_score = super(LDACustomScore, self).perplexity(X)
        return (-1) * perplexity_score

class OptimizeNLP():
    def __init__(self, filenames):
        """ Constructor
            Records corpus filename/file location code and parameters.
            Arguments:
                filenames - list of string (corpus filenames)
            Returns class instance. """
        
        """ Corpus filenames"""
        self.filenames = filenames
        
        """ Define parameters"""
        
        """standard topic number (placeholder, overwritten by grid search)"""
        self.top_n = None            
        
        """standard maximum iteration number"""
        self.MAX_ITER = 100 #1000    
        
        """search_params - dict or None (parameter values over which the LDA model is to be optimized with GridSearchCV)"""
        #self.search_params = {'n_components': [4, 6, 30], 'learning_decay': [.4, .7, .9], 'learning_offset': [5., 10., 20.]}
        #self.search_params = {'n_components': [2,3,4,5,6,7,8,9,10]} #, 'learning_decay': [.5, .7, .9], 'learning_offset': [5., 10., 20.]}
        self.search_params = {'n_components': [10, 30, 50, 75, 100, 150, 200, 300, 500]}     # wide search
        
        """TOPIC_WORD_THRESHOLD -  float (Threshold for removal of topic words (words that are present 
                                                in this share of documents are removed)"""
        self.TOPIC_WORD_THRESHOLD = 0.6 #0.4
        
        
    def read_corpus(self, filenames, path_prefix="/mnt/usb4/patentapplications"):
        """ Function to read and prepare corpus and dates etc.
            Arguments:
                path_prefix - string
                filenames - list of strings 
            Returns tuple with 4 elements:
                        - corpus as list of strings
                        - number of documents
                        - data frame corresponding exactly to corpus elements
                        - DateTime timestamps corresponding to each document in corpus"""
        """ Read file""" 
        corpus = {}
        for filename in filenames:
            # TODO: refactor to remove repetition
            if os.path_exists("{1:s}/titles/titles_{0:s}".format(filename, path_prefix)):
                filename_gz = "{1:s}/titles/titles_{0:s}".format(filename, path_prefix)
                with gzip.GzipFile(filename_gz, "r") as infile:
                    # TODO: assert that there is no overlap
                    corpus_new_entries = pickle.load(infile)
                    corpus.update(corpus_new_entries)
            if os.path_exists("{1:s}/abstracts/abstracts_{0:s}".format(filename, path_prefix)):
                filename_gz = "{1:s}/abstracts/abstracts_{0:s}".format(filename, path_prefix)
                with gzip.GzipFile(filename_gz, "r") as infile:
                    abstracts = pickle.load(infile)
                for key in abstracts.keys():
                    if key in corpus:
                        corpus[key] = corpus[key] + " \n\n\n" + abstracts[key]
                    else:
                        corpus[key] = abstracts[key]
            if os.path_exists("{1:s}/descriptions/descriptions_{0:s}".format(filename, path_prefix)):
                filename_gz = "{1:s}/descriptions/descriptions_{0:s}".format(filename, path_prefix)
                with gzip.GzipFile(filename_gz, "r") as infile:
                    descriptions = pickle.load(infile)
                for key in descriptions.keys():
                    if key in corpus:
                        corpus[key] = corpus[key] + " \n\n\n" + descriptions[key]
                    else:
                        corpus[key] = descriptions[key]
            if os.path_exists("{1:s}/claims/claims_{0:s}".format(filename, path_prefix)):
                filename_gz = "{1:s}/claims/claims_{0:s}".format(filename, path_prefix)
                with gzip.GzipFile(filename_gz, "r") as infile:
                    claims = pickle.load(infile)
                for key in claims.keys():
                    if key in corpus:
                        corpus[key] = corpus[key] + " \n\n\n" + claims[key]
                    else:
                        corpus[key] = claims[key]
            if os.path_exists("{1:s}/df/df_{0:s}".format(filename, path_prefix)):
                filename_gz = "{1:s}/df/df_{0:s}".format(filename, path_prefix)
                df_new_entries = pd.read_pickle(filename_gz, compression="gzip")        
                df = pd.concat(df, df_new_entries)
        
        df.sort_index(inplace=True)
        corpus = [corpus[dID] for dID in df.index]
        pydatetimes = df["Application date"].dt.to_pydatetime()
        NO_DOCUMENTS = len(corpus)    
        #pdb.set_trace()
        return corpus, NO_DOCUMENTS, df, pydatetimes
        
    def english_lemmatizing_stemming(self, text):
        """ Lemmatizer and stemmer based on nltk.stem.SnowballStemmer and nltk.stem.WordNetLemmatizer
            Arguments:
                text: string - text to be lemmatized and stemmed
            Returns: lemmatized and stemmed text"""
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def english_stemming_lemmatizing_preprocessing(self, text):
        """ Lemmatizing, Stemming, Preprocessing
            Arguments:
                text: string - text to be preprocessed, lemmatized and stemmed
            Returns: list of string: processed text"""
        
        result = []
        """ Preprocess"""
        for token in gensim.utils.simple_preprocess(text):
            """ Remove stop words"""
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                """ Lemmatize and stem"""
                result.append(self.english_lemmatizing_stemming(token))
        return result

    def setup_train_LDA(self, data, NO_DOCUMENTS, stemming=False, score_function='perplexity'):
        """ Function to setup and train LatentDirichletAllocation model based on scikit-learn. Parameters can be <
            optimized with GridSearchCV.
            Arguments:
                data - list of strings (Corpus)
                NO_Documents - int (number of documents in corpus)
                stemming - bool (shall the corpus be stemmed and lemmatized and preprocessed; otherwise it
                                 is taken as is) 
                score_function - string (Function to be used for GridSearchCV scores. Can be either perplexity or likelihood)
            Returns: 
                Tuple of 3 elements
                    - CountVectorizer object used for vectorization of the corpus
                    - Best LDA model (as evaluated by grid search) 
                    - Grid search CV results"""
        """ Fix python print function that cannot handle utf-8"""
        utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)
        #print(data[:5], file=utf8stdout)
        
        """ Stem corpus"""
        if stemming:
            #data2 = [self.english_stemming_lemmatizing_preprocessing(element) for element in data]
            data = [" ".join(self.english_stemming_lemmatizing_preprocessing(element)) for element in data]
        
        
        """ Vectorize corpus, Create initial bag of words"""
        if False:
            # Count vectorizer code without ngrams
            vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
            data_vectorized = vectorizer.fit_transform(data)
            
            tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
            tfidf_transformer.fit(data_vectorized)
            tfidf_transformer_vectors = tfidf_transformer.transform(vectorizer.transform(data))
            tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',)
            tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(data)
            
            """ Whith these parameters, tfidf_transformer_vectors is the same as tfidf_vectorizer_vectors. Check as follows
            
            first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
            df_t = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
            df_t.sort_values(by=["tfidf"],ascending=False)
            first_vector_count=data_vectorized[0]
            df_c = pd.DataFrame(first_vector_count.T.todense(), index=vectorizer.get_feature_names(), columns=["count"])
            df_c.sort_values(by=["count"],ascending=False)
            first_vector_ct=tfidf_transformer_vectors[0]
            df_ct = pd.DataFrame(first_vector_ct.T.todense(), index=vectorizer.get_feature_names(), columns=["count"])
            df_ct.sort_values(by=["count"],ascending=False)

            """
        
        if True:
            # tfidf code with ngrams
            vectorizer = TfidfVectorizer(use_idf=True, min_df=10, max_df=self.TOPIC_WORD_THRESHOLD, 
                                               stop_words='english', lowercase=True, 
                                               token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',
                                               ngram_range=(1, 3),
                                              )
            data_vectorized = vectorizer.fit_transform(data)

            #pdb.set_trace()
        
        """ Compute word frequencies"""
        words_freq = []
        for current_word, _ in vectorizer.vocabulary_.items():
            abs_freq = np.sum([current_word in element.split() for element in data])
            freq = abs_freq / float(len(data))
            words_freq.append((current_word, freq, abs_freq))
        
        print([(word, freq) for word, freq, abs_freq in words_freq if (freq > 0.1)])
        
        if False:
            # Fitting it again is superfluous
            """ Remove topic words"""
            #THRESHOLD = 0.2
            reduced_vocabulary = [word for word, freq, abs_freq in words_freq if (freq < self.TOPIC_WORD_THRESHOLD) and (abs_freq > 1)]
            print("Removed frequent words: {0:s}".format(str([word for word, freq, abs_freq in words_freq if freq >= self.TOPIC_WORD_THRESHOLD])))
            print("Removed infrequent words: {0:s}".format(str([word for word, freq, abs_freq in words_freq if abs_freq <= 1])))
            
            """ Redo vectorization for reduced dictionary"""
            vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', vocabulary=reduced_vocabulary, lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
            data_vectorized = vectorizer.fit_transform(data)
        
        """ Define search parameters"""     # should be done in constructir
        #if self.search_params is None:
        #    #self.search_params = {'n_components': [3, 5, 10, 15, 20], 'learning_decay': [.5, .7, .9]}
        #    #self.search_params = {'n_components': [4, 6, 10, 16], 'learning_decay': [.5, .7, .9], 'learning_offset': [5., 10., 20.]}
        #    self.search_params = {'n_components': [2,3,4,5,6,7,8,9,10]}#, 'learning_decay': [.5, .7, .9], 'learning_offset': [5., 10., 20.]}

        """ Initialize the model"""
        if score_function == 'perplexity':
            lda = LDACustomScore(max_iter=self.MAX_ITER)
        elif score_function == 'likelihood':
            lda = LatentDirichletAllocation(max_iter=self.MAX_ITER)
        else:
            assert False, "Unknown score function specified: {0:s}".format(score_function)

        """ Initialize grid search instance"""
        gv_model = GridSearchCV(lda, param_grid=self.search_params, return_train_score=True)

        """ Perform grid search"""
        gv_model.fit(data_vectorized)
        lda_model = gv_model.best_estimator_
        
        """ Print some sample output"""
        print("Best Model's Params: ", gv_model.best_params_)
        print("Best Log Likelihood Score: ", gv_model.best_score_)
        print("Model Perplexity: ", lda_model.perplexity(data_vectorized))
        print("Grid CV Results", gv_model.cv_results_)
        #pdb.set_trace()    
        
        return vectorizer, lda_model, gv_model.cv_results_

    def print_topics(self, model, vectorizer, top_p=10):
        """ Function for printing top words and frequencies for topics.
            Arguments:
                model - LDA model object
                vectorizer - CountVectorizer object used for model
                top_p - int - number of words to be printed per topic
            Returns 
                tuple: 
                    list of list of words - top words for each topic, 
                    list of list of frequencies - frequencies of top words for each topic"""
        wordlists = []
        freqlists = []
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_p - 1:-1]])
            wordlist = [(vectorizer.get_feature_names()[i]) for i in topic.argsort()[:-top_p - 1:-1]]
            wordlists.append(wordlist)
            freqlist = [topic[i] for i in topic.argsort()[:-top_p - 1:-1]]
            freqlists.append(freqlist)
        return wordlists, freqlists

    def pickle_all(self, code="F01N13_009", **kwargs):
        """ Function to pickle list of objects. Combines all optional arguments to dict which 
            is then pickled and saved.
            Arguments:
                code - string (code to be included in output filename)
                ** additional optional arguments to be included.
            Returns None"""
        with open("lda_result_" + code + ".pkl", "bw") as wfile:
            pickle.dump(kwargs, wfile, protocol=pickle.HIGHEST_PROTOCOL)

    def main(self):
        """ Main function. Handle topic modeling and output and saving of results.
            Arguments None
            Returns None"""
        
        """ Read corpus"""
        data, NO_DOCUMENTS, df, pydatetimes = self.read_corpus(self.filenames)
        document_ids = list(df.index)
        
        """ Setup, train, optimize LDA model"""
        vectorizer, lda_model, grid_cv_results = self.setup_train_LDA(data, NO_DOCUMENTS, stemming=True) 
        self.top_n = lda_model.n_components
        
        """ Print topic composition (top words and frequencies)"""
        print("LDA Model:")
        wl, fl = self.print_topics(lda_model, vectorizer)
        
        """ Obtain topic shares for each document in corpus"""
        lda_topic_shares = [lda_model.transform(vectorizer.transform([text]))[0] for text in data]
        lda_topic_shares = np.asarray(lda_topic_shares)
        lda_topic_shares = lda_topic_shares.T
        #print(lda_topic_shares[:5], '\n', len(lda_topic_shares))
        
        """ Pickle results"""
        self.pickle_all(code=code, lda_topic_shares=lda_topic_shares, lda_topic_shares_ma=lda_topic_shares_ma, pydatetimes=pydatetimes, \
                   pydatetimes_ma=pydatetimes_ma, vectorizer=vectorizer, lda_model=lda_model, \
                   wl=wl, document_ids=document_ids, grid_cv_results=grid_cv_results)
        
        """ Visualize"""
        # TODO

""" Main entry point"""

if __name__ == "__main__":
    #codes = ["F01N13_009", "C02F9_00", "F02B3_06", "Y02T10_22", "Y02T10_47"]
    #codes = ["Y02T10_47"]
    #codes = ["corpus"]
    codes = ["corpus_all"]
    for code in codes:
        ONLP = OptimizeNLP(code)
        ONLP.main()
