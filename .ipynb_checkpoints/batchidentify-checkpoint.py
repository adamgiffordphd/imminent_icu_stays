#! /Users/adamgifford/anaconda3/bin/python

import pandas as pd
import numpy as np
from customTransformers import DateTimeTransformer, DurationTransformer
import os
import datetime
from argparse import ArgumentParser
import pickle
from ediblepickle import checkpoint

FILE_PATH = './data/pickle/preproc/'
DATA_DIR = 'data/physionet.org/files/mimiciii/1.4/'
PREPROC_DIR = './data/pickle/preproc/'
DF_IDS = pd.DataFrame()

CACHE_DIR = './data/pickle/cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

def parse_arguments():
    parser = ArgumentParser()

    # mandatory arguments
    parser.add_argument('preprocfile',type=str,
                       help='local path to preprocessed database')
    parser.add_argument('batchfile',type=str,
                       help='name of csv file to batch load')
    parser.add_argument('usecols',type=list,
                       help='list of columns to load from batchfile')
    # Optional arguments
    parser.add_argument('--preproccols', type=list,
                        default=['SUBJECT_ID', 'HADM_ID','INTIME'],
                        help='list of columns to keep from preprocessed database')
    parser.add_argument('--skiprows', default=0, type=int,
                       help='number of rows to skip from start of batchfile')
    parser.add_argument('--nrows', default=100000,
                       help='number of rows per batch to read')
    parser.add_argument('--sep', default=',', help='separator')
    parser.add_argument('--header', default=0, help='header row in file')
    
    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    arguments = parser.parse_args()
    arguments.usecols = ''.join(arguments.usecols).split(',')
    if arguments.preproccols != ['SUBJECT_ID', 'HADM_ID','INTIME']:
        arguments.preproccols = ''.join(arguments.preproccols).split(',')
    # get positionals
    args = (arguments.preprocfile, arguments.batchfile, arguments.usecols)
    
    # get optionals
    kwargs = vars(arguments)
    del kwargs['preprocfile']
    del kwargs['batchfile']
    del kwargs['usecols']

    return args, kwargs

def get_meta_data(file=None):
    meta_data = {}
    # added 1 row to each to account for header=0th row
    meta_data['CHARTEVENTS.csv'] = {
        'row_count': 330712484,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','ICUSTAY_ID',
            'ITEMID','CHARTTIME','STORETIME','CGID','VALUE',
            'VALUENUM','VALUEUOM','WARNING','ERROR',
            'RESULTSTATUS','STOPPED'
        ],
        'durcolnames': ['CHART_TO_ICU']
    }
    meta_data['NOTEVENTS.csv'] = {
        'row_count': 2083181,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','CHARTDATE',
            'CHARTTIME','STORETIME','CATEGORY','DESCRIPTION',
            'CGID','ISERROR','TEXT'
        ],
        'durcolnames': ['NOTE_TO_ICU']
    }
    meta_data['LABVENTS.csv'] = {
        'row_count': 27854056,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','ITEMID',
            'CHARTTIME','VALUE','VALUENUM','VALUEUOM',
            'FLAG'
        ],
        'durcolnames': ['LAB_TO_ICU']
    }
    meta_data['PRESCRIPTIONS.csv'] = {
        'row_count': 4156451,
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

def name_fun(args,kwargs):
    ix = args[0].index('.')
    start = str(args[4])
    end = str(args[5])
    
    return 'file-' + args[0][:ix] + '-lines' + start + '-' + end + '.p'
    
@checkpoint(key=name_fun, work_dir=CACHE_DIR)
def batch_run(batchfile, batch_meta, names, usecols, skiprows,nrows,sep=',',header=0):
    skiplist=[]
    df_batch = batch_read_df(batchfile, names, usecols, skiprows,
                  nrows,sep=',',header=0)
        
    # convert time strings to datetime
    datetime_cols = [col for col in usecols if 'TIME' in col]
    datetime_cols.extend([col for col in usecols if 'DATE' in col])
    df_batch = convert_dates(df_batch,datetime_cols)

    # add to skiplist rows of df that don't have matching ['SUBJECT_ID', 'HADM_ID'] in df_ids
    toskip = list(pd.merge(df_batch, DF_IDS, on=['SUBJECT_ID', 'HADM_ID'], how="outer", indicator=True
          ).query('_merge=="left_only"').index)
    if toskip:
        skiplist.extend(toskip)

    # now merge so we can calculate when notes were taken in relation to icu admittance
    df_batch = df_batch.merge(DF_IDS,how='inner',left_on=['SUBJECT_ID', 'HADM_ID'],right_on=['SUBJECT_ID', 'HADM_ID'])

    if 'PRESCRIPTIONS' in batchfile:
        datecoltups = [('INTIME', 'STARTDATE')]
    else:
        datecoltups = [('INTIME', 'CHARTTIME')]

    df_batch = compute_durations(df_batch,datecoltups,batch_meta['durcolnames'])

    # add to skiplist rows of df where df['DAYS_NOTE_TO_ICU']<val
    if 'PRESCRIPTIONS' in batchfile:
        # prescriptions table has not time, only days
        # so verifying drugs prescribed > 1 day before
        # ICU admit only way to guarantee prescriptions
        # came before admission to ICU
        val = 1
    else:
        val = 0

    dur_cols = [col for col in df_batch.columns if any([True for dcol in batch_meta['durcolnames'] if dcol in col])]
    mask = (df_batch[dur_cols] < val).all(axis=1)
    toskip = df_batch[mask].index.tolist()
    if toskip:
        skiplist.extend(toskip)
    
    del df_batch
    return skiplist

def run_all(preprocfile,batchfile,usecols,preproccols,skiprows=0,nrows=100000,sep=',',header=0):
    global DF_IDS
    
    DF_IDS = load_preproc_df(preprocfile,preproccols)
    
    batch_meta = get_meta_data(batchfile)
    
    names = batch_meta['colnames']
    row_count = batch_meta['row_count']
    if nrows=='getall': # else must be int
        nrows = batch_meta['row_count']
    elif type(nrows) is not int:
        raise("nrows must be 'getall' or int ")
    
    skiplist = []
    cnt=0
    while skiprows<row_count:
        if skiprows + nrows > row_count:
            nrows = row_count - skiprows
        
        print('Iteration {}...'.format(cnt))
            
        skiplist.extend(batch_run(batchfile, batch_meta, names, usecols, skiprows,nrows,sep=',',header=0))
            
        skiprows += nrows
        cnt += 1
    
    return skiplist

if __name__ == "__main__":
    args, kwargs = parse_arguments()
    
    skiplist = run_all(*args,**kwargs)
    
    savesuffix = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'
    safefile = PREPROC_DIR + batchfile[:batchfile.index('.')] + '__' + savesuffix
    pickle.dump(skiplist, open(safefile, 'wb'))