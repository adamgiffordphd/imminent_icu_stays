#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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