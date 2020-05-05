# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:53:15 2020

@author: lsfer
"""

import pickle
import os

main_dir = r'C:\Users\lsfer\Desktop\etude je\factset\DataFeed_Loader\dwd'

class SaveState:
    
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.read_dir = os.path.join(main_dir, 'read')
        self.write_dir = os.path.join(main_dir, 'write')
        self.work_dir = os.path.join(main_dir, 'python_scripts')
        self.files = [f for f in os.listdir(self.read_dir)if f.endswith('.csv')]
    
    def save(self, list_objs, suffix=''):
        """
        Parameters
        ----------
        list_objs : list of objs
            List of any variables you want to save for later use.
        suffix : string, optional
            The suffix to assign to the saved object so that you don't overwrite another one. 
            The default is no suffix ('').

        """
        with open(os.path.join(self.work_dir, f'objs{suffix}.pkl'), 'wb') as f: 
            pickle.dump(list_objs, f)
        
    def restore(self, suffix=''):
        """
        Parameters
        ----------
        suffix : string, optional
            The suffix to assign to the saved object you want to restore. 
            The default is no suffix ('').
        Returns:
            The list of objects you have restored.
        """
        with open(os.path.join(self.work_dir, f'objs{suffix}.pkl'), 'rb') as f: 
            list_objs = pickle.load(f)
        return list_objs

if __name__ == '__main__':
    sv = SaveState(main_dir)
    print(sv.files)
