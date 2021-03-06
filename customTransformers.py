#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

stopWords = stopwords.words('english')
stopWords.extend(['a','b','c','d','e','f','g','h','i','j','k','l','m',
                  'n','o','p','q','r','s','t','u','v','w','x','y','z'])
stopWords.extend(['the','and','to','of','was','with','a','on','in','for',
'name','is','patient','s','he','at','as','or','one','she','his','her','am',
'were','you','pt','pm','by','be','had','your','this','date','from','there',
'an','that','p','are','have','has','h','but','o','namepattern','which','every',
'also', ',', '.', ';', '/', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
'-', '_', '=', '+', '{', '}', '[', ']', '|',':', '"', "'", '<', '>', '?'])
stopWords = set(stopWords)

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, dateus):
        self.datecols = datecols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a pandas dataframe'''        
        for col in self.datecols:
            X[col] = pd.to_datetime(X[col], format='%Y-%m-%d %H:%M:%S',
                                             errors='coerce')

        return X

class DurationTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, datetups, newcols, unit='days'):
        self.datetups = datetups
        self.newcols = newcols
        if 'second' in unit:
            self.divisor = 1
        elif 'minute' in unit:
            self.divisor = 60
        elif 'hour' in unit:
            self.divisor = 60*60
        elif 'day' in unit:
            self.divisor = 60*60*24
        elif 'week' in unit:
            self.divisor = 60*60*24*7
        elif 'month' in unit:
            self.divisor = 60*60*24*30
        elif 'year' in unit:
            self.divisor = 60*60*24*365
        else:
            raise('Invalid time unit {}'.format(unit))
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''X is a pandas dataframe.
        Subtraction is tup[0] - tup[1] for tup in self.datetups
        '''        
        for tup, col in zip(self.datetups,self.newcols):
            X[col] = (X[tup[0]] - X[tup[1]]).dt.total_seconds()/self.divisor
            
        return X
    
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
            return X[self.col_names]
#             return np.hstack([X[col].to_numpy().reshape(-1,1) for col in self.col_names]).tolist()
        else:
            if not isinstance(self.col_names,list):
                self.col_names = list(self.col_names)
                
            return [[row[col] for col in self.col_names] for row in X]

class ColumnMergeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.new_col = '_'.join([str(col) for col in self.col_names])
        return self
    
    def transform(self, X):
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        
        if type(X)==pd.core.frame.DataFrame:
            if X[self.col_names[0]].dtype=='float64':
                X.loc[:,self.new_col] = -X.loc[:,self.col_names[0]]
            else:
                X.loc[:,self.new_col] = ''
                
            for col in self.col_names:
                X.loc[:,self.new_col] = X.loc[:,self.new_col] + X.loc[:,col]
            return X
        else:
            if isinstance(X[0][0],str):
                X_merged = [[''] for row in X]
            else:
                X_merged = [[-row[0]] for row in X]
                
            X_merged = [[m_row[0] + col] for m_row,row in zip(X_merged,X) for col in row]
#             for ix,row in enumerate(X):
#                 for col in row:
#                     X_merged[ix] += col
            return X_merged
        
class EthnicityTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.ethnicites_dict = {
            'WHITE': 'WHITE',
            'WHITE - RUSSIAN': 'WHITE',
            'WHITE - OTHER EUROPEAN': 'WHITE', 
            'WHITE - EASTERN EUROPEAN': 'WHITE',
            'WHITE - BRAZILIAN': 'WHITE',
            'PORTUGUESE': 'WHITE',
            
            'BLACK/AFRICAN AMERICAN': 'BLACK',
            'BLACK/AFRICAN': 'BLACK',
            'BLACK/HAITIAN': 'BLACK',
            'BLACK/CAPE VERDEAN': 'BLACK',
            'UNKNOWN/NOT SPECIFIED': 'UNKNOWN',
            'PATIENT DECLINED TO ANSWER': 'UNKNOWN',
            'UNABLE TO OBTAIN': 'UNKNOWN',
            
            'ASIAN': 'ASIAN',
            'ASIAN - CHINESE': 'ASIAN',
            'ASIAN - VIETNAMESE': 'ASIAN',
            'ASIAN - CAMBODIAN': 'ASIAN',
            'ASIAN - FILIPINO': 'ASIAN',
            'ASIAN - KOREAN': 'ASIAN',
            'ASIAN - THAI': 'ASIAN',
            'ASIAN - JAPANESE': 'ASIAN',
            'ASIAN - OTHER': 'ASIAN',
            
            'ASIAN - ASIAN INDIAN': 'INDIAN',
            
            'OTHER': 'OTHER',
            'SOUTH AMERICAN': 'OTHER',
            'CARIBBEAN ISLAND': 'OTHER',

            'HISPANIC OR LATINO': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - DOMINICAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - SALVADORAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - COLOMBIAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - HONDURAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - CUBAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - MEXICAN': 'HISPANIC/LATINO',

            'MULTI RACE ETHNICITY': 'MULTIRACE',
            
            'MIDDLE EASTERN': 'MIDDLE EASTERN',
            
            'AMERICAN INDIAN/ALASKA NATIVE': 'AMERICAN NATIVE',
            'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 'AMERICAN NATIVE',
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'AMERICAN NATIVE'
        }
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def _transform(self,x):
        return self.ethnicites_dict[x]
    
    def transform(self,X):
        return [self._transform(x) for x in X]
            
class DiagnosisFrameTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, col_names):
        self.stopWords = stopWords
        self.regex_split = re.compile(r'[\|/;|,\s]')
        self.regex_sub1 = re.compile(r"[\|/\.-]+")
        self.regex_sub2 = re.compile(r'\s{2,}')
#             self.regex_rem1 = re.compile(r'[\d]+\b') # looks for, e.g., '01 '
        self.regex_rem1 = re.compile(r'[\d]+[a-zA-Z]*?\b')  # looks for, e.g., '01 ' and, e.g., '1st '
        self.regex_rem2 = re.compile(r'[^a-zA-Z\s]+')
        self.col_names = col_names


    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
            
    def remove_stopwords(self,d_list):
        return [d for d in d_list if d not in self.stopWords]
    
    def _transform(self,x):
        cleaned_x = []
        for row in x:
            if pd.isna(row):
                row = ''
                
            row = row.strip()
            
            row = row.replace('\\',' ')
            row = row.replace("'",' ')
            row = self.regex_rem1.sub('', row)
            row_list = self.regex_split.split(row)
            row_list = [d.strip().lower() for d in row_list]
            row_list = self.remove_stopwords(row_list)

            row = ' '.join(row_list)
            row = self.regex_sub1.sub(' ', row)
            row = self.regex_rem2.sub('', row)
            cleaned_row = self.regex_sub2.sub(' ', row)
            
            cleaned_x.append(cleaned_row)

        return cleaned_x
    
    def transform(self,X):
        for col in self.col_names:
            X[col] = self._transform(X[col])
        return X
        
class DiagnosisArrayTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.stopWords = stopWords
        self.regex_split = re.compile(r'[\|/;|,\s]')
        self.regex_sub1 = re.compile(r"[\|/\.-]+")
        self.regex_sub2 = re.compile(r'\s{2,}')
#             self.regex_rem1 = re.compile(r'[\d]+\b') # looks for, e.g., '01 '
        self.regex_rem1 = re.compile(r'[\d]+[a-zA-Z]*?\b')  # looks for, e.g., '01 ' and, e.g., '1st '
        self.regex_rem2 = re.compile(r'[^a-zA-Z\s]+')


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
            
            col = col.replace('\\',' ')
            col = col.replace("'",' ')
            col = self.regex_rem1.sub('', col)
            col_list = self.regex_split.split(col)
            col_list = [d.strip().lower() for d in col_list]
            col_list = self.remove_stopwords(col_list)

            col = ' '.join(col_list)
            col = self.regex_sub1.sub(' ', col)
            col = self.regex_rem2.sub('', col)
            cleaned_col = self.regex_sub2.sub(' ', col)
            
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
    
class EstimatorTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        # What needs to be done here?
        self.estimator = estimator
    
    def fit(self, X, y):
        # Fit the stored estimator.
        # Question: what should be returned?
        self.estimator.fit(X,y)
        return self
    
    def transform(self, X):
        # Use predict on the stored estimator as a "transformation".
        # Be sure to return a 2-D array.
        y = self.estimator.predict(X)
        return [[i] for i in y]
    
class LinearNonlinear(BaseEstimator, RegressorMixin):
    
    def __init__(self, lin, nonlin):
        self.lin = lin
        self.nonlin = nonlin
        
    # we define clones of the original models to fit the data in
    def fit(self,X,y):        
        self.lin.fit(X,y)
        y_pred = self.lin.predict(X)
        resid = y - y_pred
        self.nonlin.fit(X,resid)
            
        return self
    
    def predict(self,X):
        y = self.lin.predict(X) + self.nonlin.predict(X)
        return [[i] for i in y]