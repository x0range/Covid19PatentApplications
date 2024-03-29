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
import argparse
#import nltk 
import psutil
import resource
import pdb
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gzip
import glob

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
    def __init__(self, filenames, path_prefix, n_topics=None):
        """ Constructor
            Records corpus filename/file location code and parameters.
            Arguments:
                filenames - string or list of string (search pattern for corpus filenames or list of corpus filenames)
                path_prefix - string (path of corpus files)
            Returns class instance. """
        
        """ Corpus filenames"""
        if isinstance(filenames, list):
            self.filenames = filenames
        else:
            assert isinstance(filenames, str)
            self.filenames = glob.glob("{0:s}{1:s}{2:s}".format(path_prefix, os.sep, filenames))
        self.path_prefix = path_prefix
        
        """ Define parameters"""
        
        """standard topic number (placeholder, overwritten by grid search)"""
        self.top_n = None            
        
        """standard maximum iteration number"""
        self.MAX_ITER = 100 #1000    
        
        """search_params - dict or None (parameter values over which the LDA model is to be optimized with GridSearchCV)"""
        #self.search_params = {'n_components': [4, 6, 30], 'learning_decay': [.4, .7, .9], 'learning_offset': [5., 10., 20.]}
        #self.search_params = {'n_components': [2,3,4,5,6,7,8,9,10]} #, 'learning_decay': [.5, .7, .9], 'learning_offset': [5., 10., 20.]}
        if n_topics is None:
            self.search_params = {'n_components': [10, 30, 50, 75, 100, 150, 200, 300, 500]}     # wide search
        elif isinstance(n_topics, int):
            self.search_params = {'n_components': [n_topics]}     # no search
        else:
            assert False, "n_topics must be None or int but was: {0:s}".format(n_topics)
        
        """TOPIC_WORD_THRESHOLD -  float (Threshold for removal of topic words (words that are present 
                                                in this share of documents are removed)"""
        self.TOPIC_WORD_THRESHOLD = 0.6 #0.4
        
        
    def read_corpus(self, filenames):
        """ Function to read and prepare corpus and dates etc.
            Arguments:
                filenames - list of strings 
            Returns tuple with 4 elements:
                        - corpus as list of strings
                        - number of documents
                        - data frame corresponding exactly to corpus elements
                        - DateTime timestamps corresponding to each document in corpus"""
        """ Read file""" 
        corpus = {}
        df = None
        for filename in filenames:
            if os.path.exists(filename):
                df_new_entries = pd.read_pickle(filename, compression="gzip")        
                if df is None:
                    df = df_new_entries
                else:
                    df = pd.concat([df, df_new_entries])
                process = psutil.Process(os.getpid())
                print("Read {0:s} Corpus size: {1:7d} Memory usage: {2:8f} MB".format(filename, len(df), process.memory_info().rss/(1024**2)))
        
        """ Transform dates to standard format"""
        df["application_date"] = pd.to_datetime(df["application_date"], format="%Y%m%d")
        df["publication_date"] = pd.to_datetime(df["publication_date"], format="%Y%m%d")
        
        """ Sort by date"""
        df.sort_values(by=["application_date"], inplace=True)
        
        """ Reindex to application ID"""
        df.set_index("application_id", inplace=True)
        
        """ Join and extract corpus. This operation will require twice as much memory as the df since it 
                  copies most columns."""
        """ TODO: .astype(str) is included to avoid errors from addition with np.nan. It will translate 
                  np.nan to "nan" however. So a cleaner way would be to replace np.nan by "" manually"""
        df["corpus"] = df["title"].astype(str) + "\n\n" + df["abstract"].astype(str) +  "\n\n" + \
                                    df["description"].astype(str) +  "\n\n" + df["claims"].astype(str)
        df.drop(["title", "abstract", "description", "claims"], axis=1, inplace=True)
        
        corpus = list(df["corpus"])
        NO_DOCUMENTS = len(corpus)    
        
        df.drop(["corpus"], axis=1, inplace=True)        
        
        """ Extract datetimes"""
        pydatetimes = df["application_date"].dt.to_pydatetime()
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
        #utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)
        #print(data[:5], file=utf8stdout)
        
        """ Stem corpus"""
        if stemming:
            #data2 = [self.english_stemming_lemmatizing_preprocessing(element) for element in data]
            data = [" ".join(self.english_stemming_lemmatizing_preprocessing(element)) for element in data]
        
        
        """ Vectorize corpus, Create initial bag of words"""
        # tfidf code with ngrams
        vectorizer = TfidfVectorizer(use_idf=True, min_df=10, max_df=self.TOPIC_WORD_THRESHOLD, 
                                           stop_words='english', lowercase=True, 
                                           token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',
                                           ngram_range=(1, 3),
                                          )
        data_vectorized = vectorizer.fit_transform(data)

        """ Compute word frequencies"""
        words_freq = []
        for current_word, _ in vectorizer.vocabulary_.items():
            abs_freq = np.sum([current_word in element.split() for element in data])
            freq = abs_freq / float(len(data))
            words_freq.append((current_word, freq, abs_freq))
        
        #print([(word, freq) for word, freq, abs_freq in words_freq if (freq > 0.1)])
        
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

    def ma(self, seriesarray, windowsize):
        """ Function for moving average filter
            Arguments:
                seriesarray - array of float (input array)
                windowsize - int
            Returns: array with MA filter applied"""
        
        returnarray = None
        for series in seriesarray:
            seriessum = np.cumsum(series)
            mas = (seriessum[windowsize:] - seriessum[:-windowsize]) / windowsize
            if returnarray is None:
                returnarray = mas
            else:
                returnarray = np.vstack((returnarray, mas))
        return returnarray

    def plot_topics(self, pydatetimes, lda_topic_shares, wl, lda_topic_shares_ma=None, pydatetimes_ma=None, code=""):
        """ Function to visualize and save topic shares vs. time.
            Arguments:
                pydatetimes - list of DateTime objects (dates of documents for time axis)
                lda_topic_shares - list of list of float (shares of topics per document)
                wl - list of list of string (top words per topic)
                lda_topic_shares_ma - list of list of float (shares of topics per document after filter (moving average))
                pydatetimes_ma - list of DateTime objects (time axis for time series after applied filter)
                code - string (code to be included in output filename)
            Returns None.
                """
        
        """ Prepare empty figure of shape corresponding to number of topics"""
        ncol = 2
        nrow = np.int64((self.top_n - 1) / ncol) + 1
        fig, axs = plt.subplots(nrow, ncol, 
                                figsize=(15, 6), 
                                facecolor='w', edgecolor='k')
        axs = axs.ravel()
        
        """ Prepare code string output filenames"""
        if isinstance(code, list):
            code = "_".join(code)
        code = code.replace("*", "").replace(os.sep, "").replace(".", "_")
        
        """ Plot each topic"""
        for i in range(self.top_n):
            #pdb.set_trace()
            axs[i].set_xlim([pydatetimes_ma[0], datetime.datetime(2018, 8, 31, 0, 0)])
            axs[i].plot_date(pydatetimes, lda_topic_shares[i], marker='.')
            if lda_topic_shares_ma is not None:
                axs[i].plot_date(pydatetimes_ma, lda_topic_shares_ma[i], 'c-')
            if not i > self.top_n - 1 - ncol:
                axs[i].get_xaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig('lda_topics_detail_' + code + '.pdf')
        
        if lda_topic_shares_ma is not None:
            """ Set up second empty figure for MA plot and topic words"""
            fig1, axs1 = plt.subplots(nrow, ncol, 
                                    figsize=(15, 6), 
                                    facecolor='w', edgecolor='k')
            axs1 = axs1.ravel()
            
            axis_lim = {"xmax": [], "xmin": [], "ymax": [], "ymin": []}
            
            """ Plot each topic"""
            for i in range(self.top_n):
                axs1[i].plot_date(pydatetimes_ma, lda_topic_shares_ma[i], 'b-')
                xmin, xmax = axs1[i].get_xlim()
                ymin, ymax = axs1[i].get_ylim()
                axis_lim["xmax"].append(xmax)
                axis_lim["xmin"].append(xmin)
                axis_lim["ymax"].append(ymax)
                axis_lim["ymin"].append(ymin)

            """ Obtain common axis limits"""
            xmax = max(axis_lim["xmax"])
            xmin = min(axis_lim["xmin"])
            ymax = max(axis_lim["ymax"])
            ymin = min(axis_lim["ymin"])
            
            for i in range(self.top_n):
                """ Apply axis limits"""
                axs1[i].set_xlim(xmin, xmax)
                axs1[i].set_ylim(ymin, ymax)
                """ Add topic words to bottom of each graph"""
                axs1[i].text(0.98*xmin + 0.02*xmax, 0.9*ymin + 0.1*ymax, wl[i], bbox={'facecolor':'blue', 'alpha':0.1, 'pad':1})
                if not i > self.top_n - 1 - ncol:
                    axs1[i].get_xaxis().set_visible(False)
            fig1.tight_layout()
            fig1.savefig('lda_topics_ma_' + code + '.pdf')

    def pickle_all(self, code="*", **kwargs):
        """ Function to pickle list of objects. Combines all optional arguments to dict which 
            is then pickled and saved.
            Arguments:
                code - string (code to be included in output filename)
                ** additional optional arguments to be included.
            Returns None"""
        if isinstance(code, list):
            code = "_".join(code)
        code = code.replace("*", "").replace(os.sep, "").replace(".", "_")
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
        
        """ Apply moving average filter by time for visualization"""
        windowsize_ma = 100
        lda_topic_shares_ma = self.ma(lda_topic_shares, windowsize_ma)
        pdtma_start = np.int64(windowsize_ma / 2)
        pdtma_end = len(pydatetimes) - windowsize_ma + pdtma_start
        pydatetimes_ma = pydatetimes[pdtma_start:pdtma_end]

        """ Pickle results"""
        self.pickle_all(code=code, lda_topic_shares=lda_topic_shares, lda_topic_shares_ma=lda_topic_shares_ma, pydatetimes=pydatetimes, \
                   pydatetimes_ma=pydatetimes_ma, vectorizer=vectorizer, lda_model=lda_model, \
                   wl=wl, document_ids=document_ids, grid_cv_results=grid_cv_results)
        
        """ Visualize"""
        self.plot_topics(pydatetimes, lda_topic_shares, wl, lda_topic_shares_ma, pydatetimes_ma, code=code)

""" Main entry point"""

if __name__ == "__main__":
    
    """ Handle arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--memory",
                        help="Maximum memory size that can be used in GB",
                        type=float,
                        required=False)
    args = parser.parse_args()
    
    if args.memory:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (int(args.memory * 1024**3), hard))
    
    """ Setup and run NLP"""
    codes = ["*.pkl.gz"]
    for code in codes:
        ONLP = OptimizeNLP(code, path_prefix="../data/processed", n_topics=12)
        ONLP.main()
