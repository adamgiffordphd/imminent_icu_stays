#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

stopWords = stopwords.words('english')
stopWords.extend(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
stopWords = set(stopWords)

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, datecols):
        self.datecols = datecols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a pandas dataframe'''        
        conv_dates = []
        for col in self.datecols:
            conv_dates.append(pd.to_datetime(X[col], format='%Y-%m-%d %H:%M:%S', errors='coerce'))
            
        return np.hstack(conv_dates)


class DurationTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, datetups,unit='days'):
        self.datetups = datetups
        if unit in 'seconds':
            self.divisor = 1
        elif unit in 'minutes':
            self.divisor = 60
        elif unit in 'hours':
            self.divisor = 60*2
        elif unit in 'days':
            self.divisor = 60*2*24
        elif unit in 'weeks':
            self.divisor = 60*2*24*7
        elif unit in 'months':
            self.divisor = 60*2*24*30
        elif unit in 'years':
            self.divisor = 60*2*24*365
        else:
            raise('Invalid time unit {}'.format(unit))
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a pandas dataframe.
        Subtraction is tup[0] - tup[1] for tup in self.datetups
        '''        
        durations = []
        for tup in self.datetups:
            durations.append((X[tup[0]] - X[tup[1]]).dt.total_seconds())
            
        return np.hstack(durations)/self.divisor
    
class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        
        if type(X)==pd.core.frame.DataFrame:
            return np.hstack([X[col].to_numpy().reshape(-1,1) for col in self.col_names])
        else:
            return [[row[col] for col in self.col_names] for row in X]
    
class DiagnosisFrameTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.stopWords = stopWords
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
            
    def remove_stopwords(self,d_list):
        return [d for d in d_list if d not in self.stopWords]
    
    def _transform(self,x):
        cleaned_x = []
        for col in x:
            if pd.isna(col):
                col = ''
                
            col = col.strip()

            regex_split = re.compile(r'[\|/;|,]')
            regex_sub1 = re.compile(r"[\|/\.-]+")

            col = col.replace('\\',' ')
            col = col.replace("'",' ')
            col_list = regex_split.split(col)
            col_list = [d.strip() for d in col_list]
            col_list = self.remove_stopwords(col_list)

            col = ' '.join(col_list)
            cleaned_col = regex_sub1.sub(' ', col)
            cleaned_x.append(cleaned_col)

        return cleaned_x
    
    def transform(self,X):
        return np.hstack(np.array(list(map(self._transform, [X[:,c] for c in range(X.shape[1])]))))
    
class DiagnosisTokenizerTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
            
    def _transform(self,x):
        return word_tokenize(x)
    
    def transform(self,X):
        return list(map(self._transform, [row for row in X]))