#!/opt/anaconda3/bin/python

import pandas as pd
import numpy as np
from customTransformers import DateTimeTransformer, DurationTransformer
import os
import datetime
from argparse import ArgumentParser
import pickle
from ediblepickle import checkpoint
import multiprocessing as mp
from os import path

FILE_PATH = './data/pickle/preproc/'
DATA_DIR = 'data/physionet.org/files/mimiciii/1.4/'
PREPROC_DIR = './data/pickle/preproc/'

CACHE_DIR = './data/pickle/cache/'
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
    meta_data['NOTEEVENTS.csv'] = {
        'row_count': 2083181,
        'colnames': [
            'ROW_ID','SUBJECT_ID','HADM_ID','CHARTDATE',
            'CHARTTIME','STORETIME','CATEGORY','DESCRIPTION',
            'CGID','ISERROR','TEXT'
        ],
        'durcolnames': ['NOTE_TO_ICU']
    }
    meta_data['LABEVENTS.csv'] = {
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
    print(datecoltups)
    df[durcolnames] = dt.fit_transform(df)
    return df

def name_fun(batchfile,skiprows,nrows):
    ix = batchfile.index('.')
    start = str(skiprows)
    end = str(skiprows+nrows)
    
    name = 'file-' + batchfile[:ix] + '-lines-' + start + '-' + end + '.p'
    return name

def load_cache(filename):
    return pickle.load(open(filename, 'rb'))
    
# @checkpoint(key=name_fun, work_dir=CACHE_DIR)
def batch_run(df_ids,batchfile, batch_meta, names, usecols, skiprows,nrows,sep=',',header=0):
    filename = name_fun(batchfile,skiprows,nrows)
    savefile = CACHE_DIR + filename
    if path.exists(savefile):
        return load_cache(savefile)
    
    skiplist=[]
    df_batch = batch_read_df(batchfile, names, usecols, skiprows,
                  nrows,sep=',',header=0)
        
    # convert time strings to datetime
    datetime_cols = [col for col in usecols if 'TIME' in col]
    datetime_cols.extend([col for col in usecols if 'DATE' in col])
    df_batch = convert_dates(df_batch,datetime_cols)

    # add to skiplist rows of df that don't have matching ['SUBJECT_ID', 'HADM_ID'] in df_ids
    nomatch_df = pd.merge(df_batch, df_ids, on=['SUBJECT_ID', 'HADM_ID'], how="outer", indicator=True
              ).query('_merge=="left_only"')
    toskip = nomatch_df['ROW_ID'].to_list()
    if toskip:
        skiplist.extend(toskip)

    # now merge so we can calculate when notes were taken in relation to icu admittance
    df_batch = df_batch.merge(df_ids,how='inner',left_on=['SUBJECT_ID', 'HADM_ID'],right_on=['SUBJECT_ID', 'HADM_ID'])

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
    toskip = df_batch.ROW_ID[mask].to_list()
    if toskip:
        skiplist.extend(toskip)
    
    del df_batch
    
    pickle.dump((skiprows//nrows,skiplist), open(savefile, 'wb'))
    return skiprows//nrows, skiplist

skiplist = []
def print_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    print('Iteration {} complete.'.format(result[0]))
    skiplist.append(result[1])
    
def run_all(preprocfile,batchfile,usecols,preproccols,skiprows=0,nrows=100000,sep=',',header=0):
#     global DF_IDS
    
    df_ids = load_preproc_df(preprocfile,preproccols)
    
    batch_meta = get_meta_data(batchfile)
    
    names = batch_meta['colnames']
    row_count = batch_meta['row_count']
    if nrows=='getall': # else must be int
        nrows = batch_meta['row_count']
    elif type(nrows) is not int:
        raise("nrows must be 'getall' or int ")
        
    num_cores = mp.cpu_count()
    pool = mp.Pool(1)
    
    skiplist = []
    cnt=0
    while skiprows<row_count:
        if skiprows + nrows > row_count:
            nrows = row_count - skiprows
        
#         fargs = (df_ids,batchfile, batch_meta, names, usecols, skiprows,nrows,',',0)
#         pool.apply_async(batch_run,args=fargs,callback=print_result)
        skiplist.extend(batch_run(df_ids,batchfile, batch_meta, names, usecols, skiprows,nrows,sep=',',header=0))
            
        skiprows += nrows
        cnt += 1
        
    return skiplist
#     pool.close()
#     pool.join()

if __name__ == "__main__":
    args, kwargs = parse_arguments()
    
    skiplist=run_all(*args,**kwargs)
    batchfile = args[1]
    savesuffix = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'
    savefile = PREPROC_DIR + batchfile[:batchfile.index('.')] + '__' + savesuffix
    pickle.dump(skiplist, open(savefile, 'wb'))