#! /Users/adamgifford/anaconda3/bin/python

import pandas as pd
import numpy as np
from numpy import nansum
import os
import datetime
from argparse import ArgumentParser
import pickle
import multiprocessing as mp
from os import path
from batchidentify import (get_meta_data, load_preproc_df, 
                           batch_read_df, name_fun, load_cache,
                          )

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
    parser.add_argument('--batchfile',type=str,
                        default='NOTEEVENTS.csv',
                       help='name of csv file to batch load')
    parser.add_argument('--batchrowsfile',type=str,
                        default='NOTEEVENTS__2021_01_22_22_49_08.pkl',
                       help='name of csv file to batch load')
    parser.add_argument('--usecols',type=list,
                       help='list of columns to load from batchfile')
    # Optional arguments
    parser.add_argument('--preproccols', type=list,
                        default=['SUBJECT_ID', 'HADM_ID','ADMISSION_TYPE'],
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
    if arguments.usecols is None:
        if 'CHARTEVENTS' in arguments.batchfile:
            raise Exception('Need to figure out how to analyze CHARTEVENTS for analysis first...')
        elif 'NOTEEVENTS' in arguments.batchfile:
            arguments.usecols = ['ROW_ID','SUBJECT_ID','HADM_ID','CATEGORY','TEXT']
        elif 'LABVENTS' in arguments.batchfile:
            raise Exception('Do not use LABEVENTS for analysis')
        else:
            arguments.usecols = []
    
    
    if arguments.preproccols != ['SUBJECT_ID', 'HADM_ID','ADMISSION_TYPE']:
        arguments.preproccols = ''.join(arguments.preproccols).split(',')
    # get positionals
    args = (arguments.preprocfile, arguments.batchfile, arguments.batchrowsfile, arguments.usecols, arguments.preproccols)
    
    # get optionals
    kwargs = vars(arguments)
    del kwargs['preprocfile']
    del kwargs['batchfile']
    del kwargs['batchrowsfile']
    del kwargs['usecols']
    del kwargs['preproccols']

    return args, kwargs

# def print_result(result):
#     # This is called whenever foo_pool(i) returns a result.
#     # result_list is modified only by the main process, not the pool workers.
#     print('Iteration {} complete.'.format(result[0]))
#     DF = DF + result[1]
    
def batch_run(df_ids,batchfile, batch_rowids, batch_meta, names, usecols, skiprows,nrows,sep=',',header=0):
    filename = name_fun(batchfile,skiprows,nrows)
    savefile = CACHE_DIR + filename
    if path.exists(savefile):
        return load_cache(savefile)
    
    df_batch = batch_read_df(batchfile, names, usecols, skiprows,
                  nrows,sep=',',header=0)
    df_batch = df_batch[df_batch['ROW_ID'].isin(batch_rowids)]
    
    df_merged = df_ids.merge(df_batch, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'],right_on=['SUBJECT_ID', 'HADM_ID'])
#     print(df_ids.columns)
    
    if len(df_merged)==0:
        df_byadmit_type = pd.DataFrame({
            ('NULL_TEXT','nansum'): [0,0,0,0],
            ('ADMISSION_TYPE','count'): [0,0,0,0],
        }, index=['ELECTIVE', 'EMERGENCY', 'NEWBORN', 'URGENT'])
    else:
        df_merged['NULL_TEXT'] = df_merged['TEXT'].isnull()
        df_byadmit_type = df_merged.groupby(['ADMISSION_TYPE']).agg({'NULL_TEXT': [nansum],
                                              'ADMISSION_TYPE': 'count'})
    
    pickle.dump((skiprows//nrows,df_byadmit_type), open(savefile, 'wb'))
    return skiprows//nrows, df_byadmit_type

def run_all(preprocfile,batchfile,batchrowsfile,usecols,preproccols,skiprows=0,nrows=100000,sep=',',header=0):
    DF = pd.DataFrame({
            ('NULL_TEXT','nansum'): [0,0,0,0],
            ('ADMISSION_TYPE','count'): [0,0,0,0],
        }, index=['ELECTIVE', 'EMERGENCY', 'NEWBORN', 'URGENT'])
    
    
    df_ids = load_preproc_df(preprocfile,preproccols)
    batch_rowids = load_preproc_df(batchrowsfile)
    
    batch_meta = get_meta_data(batchfile)
    
    names = batch_meta['colnames']
    row_count = batch_meta['row_count']
        
#     num_cores = mp.cpu_count()
#     pool = mp.Pool(num_cores)
    
    cnt=0
    while skiprows<row_count:
        if skiprows + nrows > row_count:
            nrows = row_count - skiprows
        
#         fargs = (df_ids,batchfile, batch_rowids[cnt], batch_meta, names, usecols, skiprows,nrows,',',0)
#         pool.apply_async(batch_run,args=fargs,callback=print_result)
        ix,batch_df = batch_run(df_ids,batchfile, batch_rowids[cnt], batch_meta, names, usecols, skiprows,nrows,sep=',',header=0)
        DF = DF.add(batch_df, fill_value=0)
        
        skiprows += nrows
        cnt += 1
        
    return DF
#     pool.close()
#     pool.join()

if __name__ == "__main__":
    args, kwargs = parse_arguments()
    
    DF = run_all(*args,**kwargs)
    batchfile = args[1]
    savesuffix = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'
    savefile = PREPROC_DIR + 'stats__' + batchfile[:batchfile.index('.')] + '__' + savesuffix
    pickle.dump(DF, open(savefile, 'wb'))