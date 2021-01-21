#!/usr/bin/env python

import pandas as pd
import numpy as np
from customTransformers import DateTimeTransformer, DurationTransformer
from ediblepickle import checkpoint
import os
import datetime

FILE_PATH = './data/pickle/preproc/'
DATA_DIR = 'data/physionet.org/files/mimiciii/1.4/'
PREPROC_DIR = './data/pickle/preproc/'

CACHE_DIR = './data/pickle/cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

def get_meta_data(file=None):
    meta_data = {}
    # added 1 row to each to account for header=0th row
    meta_data['CHARTEVENTS.csv'] = {
        'nrows': 330712484,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','ICUSTAY_ID',
            'ITEMID','CHARTTIME','STORETIME','CGID','VALUE',
            'VALUENUM','VALUEUOM','WARNING','ERROR',
            'RESULTSTATUS','STOPPED'
        ],
        'durcolnames': ['CHART_TO_ICU']
    }
    meta_data['NOTEVENTS.csv'] = {
        'nrows': 2083181,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','CHARTDATE',
            'CHARTTIME','STORETIME','CATEGORY','DESCRIPTION',
            'CGID','ISERROR','TEXT'
        ],
        'durcolnames': ['NOTE_TO_ICU']
    }
    meta_data['LABVENTS.csv'] = {
        'nrows': 27854056,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','ITEMID',
            'CHARTTIME','VALUE','VALUENUM','VALUEUOM',
            'FLAG'
        ],
        'durcolnames': ['LAB_TO_ICU']
    }
    meta_data['PRESCRIPTIONS.csv'] = {
        'nrows': 4156451,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','ICUSTAY_ID',
            'STARTDATE','ENDDATE','DRUG_TYPE','DRUG',
            'DRUG_NAME_POE','DRUG_NAME_GENERIC',
            'FORMULARY_DRUG_CD','GSN','NDC','PROD_STRENGTH',
            'DOSE_VAL_RX','DOSE_UNIT_RX','FORM_VAL_DISP',
            'FORM_UNIT_DISP','ROUTE'
        ],
        'durcolnames': ['DRUG_TO_ICU']
    }
    
    if file:
        return meta_data[file]
    else:
        return meta_data

def load_preproc_df(filename,colnames):
    '''colnames should be list to always return dataframe
    rather than series
    '''
    df = pd.read_pickle(FILE_PATH + filename)
    return df[colnames]

def checkpoint_name(args, kwargs):
    ix = args[0].index('.')
    
    # make it so start:end is inclusive, inclusive for easy viewing
    start = str(kwargs[0])
    end = str(kwargs[0]+kwargs[1]-1)
    
    return args[:ix] + '_rows_' + start + '_to_' + end + '.p'

@checkpoint(key=checkpoint_name, work_dir=CACHE_DIR)
def batch_read_df(filename, names, usecols='all', skiprows=0,
                  nrows=100000,sep=',',header=0):
    if usecols=='all':
        usecols = names
    
    df = pd.read_csv(DATA_DIR + filename, sep=sep, header=header, names=names,
                     skiprows=skiprows, nrows=nrows, usecols=usecols)
    return df

def convert_dates(df,datecols):
    dtf = DateTimeTransformer(datecols)
    df[datecols] = dtf.fit_transform(df)
    return df

def compute_durations(df,datecoltups,durcolnamesbase,unit='days'):
    dt = DurationTransformer(datecoltups,unit)
    
    durcolnames = [unit.upper() + '_' + base for base in durcolnamesbase]
    df[durcolnames] = dt.fit_transform(df)
    return df

def run_all(preprocfile,preproccols,batchfile,usecols,skiprows=0,nrows='getall',sep=',',header=0):
    df_ids = load_preproc_df(preprocfile,preproccols)
    
    batch_meta = get_meta_data(batchfile)
    
    names = batch_meta['colnames']
    if nrows=='getall': # else must be int
        nrows = batch_meta['nrows']
    elif type(nrows) is not int:
        raise("nrows must be 'getall' or int ")
    
    skiplist = []
    cnt=0
    while skiprows<row_count:
        if skiprows + nrows > row_count:
            nrows = row_count - skiprows
        
        print('Iteration {}...'.format(cnt))
        df_batch = batch_read_df(batchfile, names, usecols, skiprows,
                  nrows,sep=',',header=0)
        
        # convert time strings to datetime
        datetime_cols = [col for col in names 
                         if 'TIME' in col or if 'DATE' in col]
        df_batch = convert_dates(df_batch,datetime_cols)
        
        # add to skiplist rows of df that don't have matching ['SUBJECT_ID', 'HADM_ID'] in df_ids
        toskip = list(pd.merge(df_batch, df_ids, on=['SUBJECT_ID', 'HADM_ID'], how="outer", indicator=True
              ).query('_merge=="left_only"').index)
        if toskip:
            skiplist.extend(toskip)
            
        # now merge so we can calculate when notes were taken in relation to icu admittance
        df_batch = df_batch.merge(df_ids,how='inner',left_on=['SUBJECT_ID', 'HADM_ID'],right_on=['SUBJECT_ID', 'HADM_ID'])
        
        if 'PRESCRIPTIONS' in batchfile:
            datecoltups = [('INTIME', 'STARTDATE')]
        else:
            datecoltups = [('INTIME', 'CHARTTIME')]
            
        df_batch = compute_durations(df_batch,datecoltups,batch_meta['durcolnames'])
        
        # add to skiplist rows of df where df['DAYS_NOTE_TO_ICU']<0
        dur_cols = [col for col in df_batch.columns if batch_meta['durcolnames'] in col]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        toskip = df_batch.index[df_batch[dur_cols] < 0].tolist() # --> fix her for list of cols dur_cols condition
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        if toskip:
            skiplist.extend(toskip)
            
        skiprows += nrows
        cnt += 1
    
    return skiplist

if __name__ == "__main__":
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # argument parser to get inputs from command line
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    skiplist = run_all(preprocfile,preproccols,batchfile,usecols,skiprows=0,nrows='getall',sep=',',header=0)
    
    savesuffix = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'
    safefile = PREPROC_DIR + batchfile[:batchfile.index('.')] + '__' savesuffix
    dill.dump(skiplist, open(safefile, 'wb'))