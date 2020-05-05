# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:32:45 2020

@author: lsfer
"""


import os, gc
import pandas as pd
from sentiment import SentimentAnalysis
from opinionfact import FactOpinion
from utils import main_dir
from acp import prepare_acp, plot_acp


class Classify(FactOpinion, SentimentAnalysis):
    
    def __init__(self, main_dir):
        FactOpinion.__init__(self, main_dir)
        SentimentAnalysis.__init__(self, main_dir)
        self.head = self.columns + list(self.entity_dict.keys()) + list(self.tag_dict.keys()) + list(self.dep_dict.keys())
        
    def __call__(self, files=None, verbose=True):
        if verbose:
            print(f'\nProcessing files...')
            print(f'\nProcessing sentiment...')
        SentimentAnalysis.__call__(self, files=files)
        if verbose:
            print(f'\nProcessing opinion fact...')
        FactOpinion.__call__(self, files=files)
        gc.collect()
        if verbose:
            print('\nDone !')
    
    def acp(self, file, n_components=None, svd_solver='auto', rebuild=False, verbose=True):
        if verbose:
            print('\nPreparing for Principal Components Analysis...')
        if rebuild:
            self.__call__(files=[file])
        db = pd.read_csv(os.path.join(self.write_dir, file), sep=';')[self.head]
        plot_acp(
            X_pca = prepare_acp(db, n_components=n_components, svd_solver=svd_solver),
            path_save = os.path.join(self.write_dir, f'acp_{n_components}.png')
                )
        del db
        gc.collect()
        if verbose:
            print('\nPCA Done !')
     


if __name__ == '__main__':
    cl = Classify(main_dir)
    cl()
    cl.acp(file='test_file.csv', n_components=2)
