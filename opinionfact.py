# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:10:57 2020

@author: lsfer
"""
from utils import main_dir, SaveState
from copy import deepcopy as dc
from tqdm import tqdm
import pandas as pd
import spacy
import os
nlp = spacy.load('en_core_web_sm')


class FactOpinion(SaveState):
    
    def __init__(self, main_dir):
        super(FactOpinion, self).__init__(main_dir)
        self.tag_dict = {
        '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0, 
        '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
        'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 
        'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0, 
        'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0, 
        'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0, 
        'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
        'OOV': 0, 'TRAILING_SPACE': 0}
        self.entity_dict = {
        'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
        'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
        'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
        'ORDINAL': 0, 'CARDINAL': 0 }
        self.dep_dict = {
        'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
        'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
        'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
        'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
        'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
        'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
        'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
        'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
        'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0, 
        'TRAILING_DEP': 0}
        
    def prepare_sentence(self, sent):
        if type(sent) == str:
            sent = [s for s in nlp(sent).sents]
        else:
            try:
                return sum([self.prepare_sentence(x) for x in sent], [])
            except :
                return []
        return sent
    
    def process_sentence(self, sent):
        sentences_with_features = pd.DataFrame(columns=['sentence'])
        sentences = self.prepare_sentence(sent)
        for i, sentence in enumerate(sentences):
            sentence_with_features = {}
            # Pure sentence
            sentence_with_features['sentence'] = str(sentence)
            # Entities found, categoried by their label
            entities_dict = self.number_of_specific_entities(sentence)
            sentence_with_features.update(entities_dict)
            # Parts of speech of each token in a sentence and their amount
            pos_dict = self.number_of_fine_grained_pos_tags(sentence)
            sentence_with_features.update(pos_dict)
            # Dependencies tags for each token in a sentence
            dep_dict = self.number_of_dependency_tags(sentence)
            sentence_with_features.update(dep_dict)
            sentences_with_features = sentences_with_features.append(pd.DataFrame(sentence_with_features, index=[i]))
        return sentences_with_features
    
    def number_of_fine_grained_pos_tags(self, sent):
        """
        Find all the tags related to words in a given sentence. Slightly more
        informative than part of speech tags, but overall similar data.
        Only one might be necessary. 
        For complete explanation of each tag, visit: https://spacy.io/api/annotation
        """
        tag_dict = dc(self.tag_dict)
        
        for token in sent:
            if token.is_oov:
                tag_dict['OOV'] += 1
            elif token.tag_ == '':
                tag_dict['TRAILING_SPACE'] += 1
            else:
                tag_dict[token.tag_] += 1
                
        return tag_dict
    
    def number_of_specific_entities(self, sent):
        """
        Finds all the entities in the sentence and returns the amont of 
        how many times each specific entity appear in the sentence.
        """
        entity_dict = dc(self.entity_dict)
        
        entities = [ent.label_ for ent in sent.as_doc().ents]
        for entity in entities:
            entity_dict[entity] += 1
            
        return entity_dict
    
    def number_of_dependency_tags(self, sent):
        """
        Find a dependency tag for each token within a sentence and add their amount
        to a dictionary, depending how many times that particular tag appears.
        """
        dep_dict = dc(self.dep_dict)
        
        for token in sent:
            if token.dep_ == '':
                dep_dict['TRAILING_DEP'] += 1
            else:
                try:
                    dep_dict[token.dep_] += 1
                except:
                    print('Unknown dependency for token: "' + token.orth_ +'". Passing.')
            
        return dep_dict
    
    def opfact(self, f):
        opfact = [self.process_sentence(text).drop(columns=['sentence'], inplace=False).sum(axis=0).to_frame().transpose() for text in f['text']]
        if len(opfact) > 0:
            opfact = pd.concat(opfact, axis=0, ignore_index=True)
        else:
            opfact = pd.DataFrame()
        return opfact
    
    def __call__(self, files=None):
        if files is None:
            files = self.files
        for f in tqdm(files):
            try:
                file = pd.read_csv(os.path.join(self.write_dir, f), sep=';')
            except:
                file = pd.read_csv(os.path.join(self.read_dir, f), sep=';')
            opf = self.opfact(file)
            pd.concat([file, opf], axis=1).to_csv(os.path.join(self.write_dir, f), sep=';', index=False)
    
if __name__ == '__main__':
    fo = FactOpinion(main_dir)
    t = r'Hello, world. How are you doing today ? !'
    print(t)
    print(fo.process_sentence(t))
    fo()