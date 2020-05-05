# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:49:37 2020

@author: lsfer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:01:57 2020

@author: lsfer
"""

import os
from utils import SaveState, main_dir
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk.tokenize import word_tokenize
from copy import deepcopy as dc
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
breakdown = swn.senti_synset('breakdown.n.03')
import pandas as pd
import numpy as np
from tqdm import tqdm
import string
from datetime import datetime as dt
from matplotlib import cm, pyplot as plt
from matplotlib.colors import ListedColormap

class SentimentAnalysis(SaveState):
    def __init__(self, main_dir, lim=None):
        super(SentimentAnalysis, self).__init__(main_dir)
        if lim is not None:
            self.files = self.files[:lim]
        self.filtre = [wn.NOUN, wn.ADJ, wn.ADV, wn.VERB]
        self.cols = ['pos_score', 'neg_score', 'self_score']
        self.w_cols = [col.replace('score', 'weight') for col in self.cols]
        self.all_cols = [u for t in self.cols for u in [t, t.replace('score', 'weight')]]
        self.columns = dc(self.all_cols)
        self.columns.append('len_speak')
        self.selfref = ['our', 'us', 'we', 'ourselves', 'I', 'i', 'my', 'myself', 'me']
        a = cm.get_cmap('tab10', 10)
        self.colors_all = a(range(len(self.cols)))
        self.colors = {'all': ListedColormap(self.colors_all)}
        for i, column in enumerate(self.cols):
            self.colors[column] = self.colors_all[i, :]
        
    @staticmethod
    def conv_date(s):
        try:
            date = dt.strptime(s, "%d/%m/%Y")
        except:
            date = dt.strptime(s, "%Y-%m-%d")
        return date
              
    
    @staticmethod
    def penn_to_wn(tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        else:
            return None
        
    def get_sentiment(self, word, tag):
        """ 
        returns list of pos neg and objective score. 
        But returns empty list if not present in senti wordnet. 
        """
        wn_tag = self.penn_to_wn(tag)
        if wn_tag not in self.filtre:
            return []
    
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            return []
    
        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            return []
    
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        return [swn_synset.pos_score(),swn_synset.neg_score()]#,swn_synset.obj_score()
    
    @staticmethod
    def robustify(text=''):
        if type(text) != str:
            try:
                text = str(text)
            except:
                text = ''
        return text

    def score(self, text=''):
        """

        Parameters
        ----------
        text : string, doesn't accept dataframe, list, or any other type.
            The text you want to score for sentiments. The default is ''.

        Returns
        -------
        p_score : float
            positive score, normalized as a function of the length of text.
            To get the full positive scoring of text, you just need to return p_score * pn_weight
        pn_weight : int
            sentiment weight, which is a function of the length of text. It is the same value
            for positive and negative sentiment.
        n_score : float
            negative score, normalized as a function of the length of text.
            To get the full negative scoring of text, you just need to return n_score * pn_weight
        pn_weight : int
            sentiment weight, which is a function of the length of text. It is the same value
            for positive and negative sentiment.
        self_score : float
            The scoring of self-references in the text. 
            Only relevant if you have a first person speech.
            To get the full self-scoring of text, you just need to return self_score * self_weight
        self_weight : int
            self-references weight, as a function of the length of the text. 
            It is always greater than the pn_score.
        
        Warning: 
            an empty text returns weights of 0, but a short text could also get weights of 0.

        """
        text = self.robustify(text)
        p_score, n_score, self_score = 0, 0, 0
        for dot in string.punctuation:
            text = text.replace(dot,'')
        tokenized_text = word_tokenize(text)
        self_weight = len(tokenized_text)
        pn_weight = len(tokenized_text)
        if len(tokenized_text) > 0:
            self_score = round(np.mean([w in self.selfref for w in tokenized_text])*100, 2)
            stemmed_text = [ps.stem(x) for x in tokenized_text]
            tags = pos_tag(stemmed_text)
            senti_val = [(x.lower(), self.get_sentiment(x,y)) for (x,y) in tags]
            senti_val = list(filter(lambda x : len(x[1])>0, senti_val))
            pn_weight = len(senti_val)
            if len(senti_val) > 0:
                p_score = round(np.mean([t[0] for a,t in senti_val])*100, 2)
                n_score = round(np.mean([t[1] for a,t in senti_val])*100, 2)
        return p_score, pn_weight, n_score, pn_weight, self_score, self_weight

    @staticmethod
    def get_all_text(df):
        """

        Parameters
        ----------
        df : dataframe
            Must contain a column 'text' which contains rows of text.

        Returns
        -------
        t : string
            The concatenation of all texts in the 'text' column of the dataframe, lowercased.
            In case it is not possible, returns an empty string.

        """
        try:
            t = ' '.join(df['text'].values).lower()
        except:
            t = ''
        return t
    
    def len_speak(self, f):
        f['len_speak'] = f[self.w_cols].sum(axis=1)
        return f

    def __call__(self, filtre=None, files=None):
        """

        Parameters
        ----------
        filtre : list of wordnet tags, optional
            If you want to specify a particular list of wordnet tags to do the sentiment analysis. 
            The default is None, will take the default value specified in the __init__.
        files : list of strings, optional
            The files you want to analyze. 
            The default is None, will take the default value specified in the __init__.
            The function supposes that each file is a csv file, 
            with a dataframe that has at least one 'text' column containing texts to process.

        Assigns the sentiment scores to all files to process.

        """
        if filtre is not None:
            self.filtre = filtre
        files = files or self.files
        for test_file in tqdm(files):
            self.assign_scores(test_file)


    def assign_scores(self, test_file):
        """

        Parameters
        ----------
        test_file : string
            Name of a file.

        Returns
        -------
        df : Dataframe
            Dataframe with additionnal columns obtained by wrapping score on each text row.

        """
        df = pd.read_csv(os.path.join(self.read_dir, test_file), sep=';')
        df[self.all_cols] = pd.DataFrame(zip(*df['text'].apply(self.score)), index=self.all_cols).transpose()
        df = self.len_speak(df)
        df.to_csv(os.path.join(self.write_dir, test_file), index=False, sep=';') 
        return df
    
    def analyze(self, suffix=dt.now().strftime("%d-%m-%Y_%H-%M-%S"), precision=200, **syns): 
        """

        Parameters
        ----------
        suffix : string, optional
            Suffix to assign when saving figure to not overwrite previous figures. 
            The default is current date and time.
        precision : int, optional
            The nb of bins for the histogram. The default is 200.
        **syns : name = name_of_file
            The files you want to compare with each other.
            For each you have to name the argument, which will be the name displayed in the legend
            of the histogram.

        Returns
        -------
        Creates and saves histograms, with median vertical line of each distribution.
        
        Warning : This function requires that you have already run a __call__ for the given files.

        """
        for s_name, s_val in syns.items():
            syns[s_name] = pd.read_csv(os.path.join(self.write_dir, s_val), sep=';')[self.all_cols]
        for col in self.cols:
            for s_name, s_val in syns.items():
                q_scores = dc(s_val)
                q_scores.fillna(0, inplace=True)
                q_scores, w = q_scores[col].values, q_scores[col.replace('score', 'weight')].values 
                plt.hist(q_scores, weights=w, bins=precision, density=True, label=s_name)#, color=self.colors[col]
                plt.axvline(x=np.quantile(a=q_scores, q=0.5), linewidth=1)
            plt.legend(loc='upper right')
            plt.title(f"Répartition des {col}")
            plt.savefig(os.path.join(self.write_dir, f'{col}_{suffix}.png'))
            plt.show()  
    
    def spread(self, title='', **list_df):
        """
        Parameters
        ----------
        list_df : list of dataframes with name as name of the argument
            The files you want to compare. The name will be displayed in the label.
        title : string, optional
            Custom name. The default is ''.


        """
        for df_name, df_val in list_df.items():
            list_df[df_name] = pd.read_csv(os.path.join(self.write_dir, df_val), sep=';')[self.all_cols]
        x = np.zeros((len(self.cols), len(list_df)))
        eps = 0.5/len(list_df)
        for i, col in enumerate(self.cols):
            for j, df in enumerate(list_df.values()):
                x[i,j] = np.average(df[col].values, weights=df[col.replace('score', 'weight')].values)
        x = np.transpose(x)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        X = np.arange(len(self.cols))
        s=0
        for i in range(len(list_df)):
            ax.bar(X + s*eps, x[i], width = eps)
            s+=1
        plt.xticks(ticks=X+0.25, labels=self.cols)
        ax.legend(labels=list_df.keys(), loc='upper right')
        ax.set_title(f'{title} comparison of scores')
        plt.savefig(os.path.join(self.write_dir, f'{title} comparison of scores.png'))
        plt.show()
    
    def serie_temp(self, file):
        """

        Parameters
        ----------
        file : string
            name of file.
            Supposes that there is a time succession in the rows of text.

        """
        f = pd.read_csv(os.path.join(self.write_dir, file), sep=';')
        for col in self.cols:
            series = f[col].values
            nums = f['self_weight'].values.astype(int) 
            plotting = []
            for s,n in zip(series, nums):
                plotting += [s]*n
            plt.plot(plotting, label=col, color=self.colors[col])
        plt.legend(loc='upper right')
        plt.title(f'Temporal evolution of sentiment')
        plt.savefig(os.path.join(self.write_dir, f'Temporal evolution of {file[:-4]}.png'))
        plt.show()
                
    def hist(self, file, precision=20):   
        f = pd.read_csv(os.path.join(self.write_dir, file), sep=';')
        for col in self.cols:                
            series = f[col].values
            nums = f[col.replace('score', 'weight')].values.astype(int)
            plt.hist(series, weights=nums, label=col, color=self.colors[col], bins=precision, density=True)
            plt.legend(loc='upper right')
            plt.title(f'Histogramm of {col}')
            plt.savefig(os.path.join(self.write_dir, f'Histogramm of {col} for {file[:-4]}.png'))
            plt.show()
  
    @staticmethod
    def KL(a, b):
        d = {}
        for k, v in a.items():
            v = np.asarray(v[0], dtype=np.float)
            vv = np.asarray(b[k][0], dtype=np.float)
            kl = np.sum(np.where(v != 0, v * np.log(v / vv), 0))
            d[k] = kl
            print(f'For {k} the KL divergence is {kl}')
        return d



if __name__ == '__main__':
    SA = SentimentAnalysis(main_dir)
    SA() 
    SA.analyze(file1='test_file.csv', file2='test_file.csv')
    SA.spread(file1='test_file.csv', file2='test_file.csv')
    SA.serie_temp('test_file.csv')
    SA.hist('test_file.csv')


