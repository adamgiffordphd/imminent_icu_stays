#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import multiprocessing as mp
from pathlib import Path
from os import path
import pickle
import datetime
"""
note: this represents 90% of code used, rest is part of
jupyter notebook "assets"
"""
def run_chartevents_and_icustays():
    """
    this does all of the prework setup to merge the 4 datasets based on
    subject id, hospital admission id, and icu stay id. it also calculates
    timing of events relative to each other, and identifies the pairwise
    prescribed drug combinations found in the dataset for use in analyzing
    the relationship between pairs of drugs and time to icu stay
    
    this is a quick-and-dirty copy and paste from the jupyter notebook, so
    a lot of things are going on in this function. ideally, I would group
    lines by objective into separate functions, but I wanted to make sure
    I was able to submit before the deadline!
    """
    #1. Load patients
    # import patient info
    data_dir = 'data/physionet.org/files/mimiciii/1.4/'
    patient_file = 'PATIENTS.csv'
    df_patients = pd.read_csv(data_dir + patient_file)
    
    df_patients=df_patients.drop(['ROW_ID','DOD_HOSP','DOD_SSN'],axis=1)

    # convert date strings to datetime
    df_patients.DOB = pd.to_datetime(df_patients.DOB,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_patients.DOD = pd.to_datetime(df_patients.DOD,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # import admissions info
    admissions_file = 'ADMISSIONS.csv'
    df_admissions = pd.read_csv(data_dir + admissions_file)
    df_admissions = df_admissions.drop(['ROW_ID','RELIGION','LANGUAGE','MARITAL_STATUS','ETHNICITY'],axis=1)
    
   # convert time strings to datetime
    df_admissions.ADMITTIME = pd.to_datetime(df_admissions.ADMITTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_admissions.DISCHTIME = pd.to_datetime(df_admissions.DISCHTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_admissions.EDREGTIME = pd.to_datetime(df_admissions.EDREGTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_admissions.EDOUTTIME = pd.to_datetime(df_admissions.EDOUTTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # calculate length of hospital stay
    df_admissions['HOSPITAL_DAYS'] = (df_admissions['DISCHTIME'] - df_admissions['ADMITTIME']).dt.total_seconds()/(24*60*60)
    # negative admit days = dead on arrival, remove
    doa_idx = df_admissions[df_admissions['HOSPITAL_DAYS']<0].index
    df_admissions = df_admissions.drop(doa_idx,axis=0)
    
    # merge patient and admissions df
    df_patient_admit = df_patients.merge(df_admissions,how='inner',left_on=['SUBJECT_ID'],right_on=['SUBJECT_ID'])
    
    # calculate age at admit
    df_patient_admit['ADMIT_AGE'] = df_patient_admit['ADMITTIME'].dt.year - df_patient_admit['DOB'].dt.year
    
    # 2. Remove patients <age
    # not necessary, but wanted to limit analysis to non-pediatric issues
    age = 10
    child_idx = df_patient_admit[df_patient_admit['ADMIT_AGE']<age].index
    child_patients = df_patient_admit.iloc[child_idx]['SUBJECT_ID'].unique()
    df_patient_admit = df_patient_admit.drop(child_idx, axis=0)
    
    # 3. Load icustays
    # import icu stays info
    icustays_file = 'ICUSTAYS.csv'
    df_icustays = pd.read_csv(data_dir + icustays_file)
    
    child_idx = df_icustays[df_icustays['SUBJECT_ID'].isin(child_patients)].index
    df_icustays = df_icustays.drop(child_idx,axis=0)
    
    df_icustays = df_icustays.drop(['ROW_ID'],axis=1)
    df_icustays.INTIME = pd.to_datetime(df_icustays.INTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_icustays.OUTTIME = pd.to_datetime(df_icustays.OUTTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    #join admissions and icu stays
    df_admit_icu = df_admissions.merge(df_icustays,how='inner',left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'])
    
    # removing columns not used for further analyses
    cols_to_rmv = list(set(df_admit_icu.columns) & set(df_patient_admit))
    # keep SUBJECT_ID and HADM_ID for merge
    sbj_idx = cols_to_rmv.index('SUBJECT_ID')
    cols_to_rmv.pop(sbj_idx)
    h_idx = cols_to_rmv.index('HADM_ID')
    cols_to_rmv.pop(h_idx)
    
    df_admit_icu = df_admit_icu.drop(cols_to_rmv,axis=1)
    
    df_patient_admit_icu = df_patient_admit.merge(df_admit_icu,how='inner',left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'])
    
    # calculate days from hospital admission to icu admission
    df_patient_admit_icu['DAYS_ADM_TO_ICU'] = (df_patient_admit_icu['INTIME'] - df_patient_admit_icu['ADMITTIME']).dt.total_seconds()/(24*60*60)

    # 4. Load prescriptions
    # import prescriptions info
    prescrips_file = 'PRESCRIPTIONS.csv'
    df_prescrips = pd.read_csv(data_dir + prescrips_file,low_memory=False)
    
    df_prescrips = df_prescrips.drop(['ROW_ID','GSN','DRUG','DRUG_NAME_POE','DRUG_NAME_GENERIC','FORMULARY_DRUG_CD'],axis=1)
    ndc_nan_idx = df_prescrips[df_prescrips['NDC'].isna()].index
    df_prescrips=df_prescrips.drop(ndc_nan_idx,axis=0)
    
    df_prescrips.STARTDATE = pd.to_datetime(df_prescrips.STARTDATE,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # using NDC code for drug identity rather than drug names since working with numerics is easier
    df_prescrips['NDC'] = df_prescrips['NDC'].astype('int64')
    
    child_idx = df_prescrips[df_prescrips['SUBJECT_ID'].isin(child_patients)].index
    df_prescrips = df_prescrips.drop(child_idx,axis=0)
    
    # merge datasets
    # keep left merge here because we want to be flexible for including patients admitted to ICU w/o prescription info
    df_patient_admit_icu_prescrip = df_patient_admit_icu.merge(df_prescrips,how='left',left_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],right_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    
    # calculate days from hospital admission to drug prescription
    df_patient_admit_icu_prescrip['DAYS_ADM_TO_DRUG'] = (df_patient_admit_icu_prescrip['STARTDATE'] - df_patient_admit_icu_prescrip['ADMITTIME']).dt.total_seconds()/(24*60*60)
    
    # calculate time of drug prescription relative to ICU stay (<0 means ICU stay comes before drug prescription)
    df_patient_admit_icu_prescrip['DAYS_DRUG_BEFORE_ICU'] = df_patient_admit_icu_prescrip['DAYS_ADM_TO_ICU'] - df_patient_admit_icu_prescrip['DAYS_ADM_TO_DRUG']
    
    # keep only rows where drugs were prescribed prior to ICU stay
    df_patient_admit_icu_prescrip_drugsfirst = df_patient_admit_icu_prescrip[df_patient_admit_icu_prescrip['DAYS_DRUG_BEFORE_ICU']>0]

    # let's make a simplified binary classification of whether the ICU stay occurred within 24 hours of hospital admission
    df_patient_admit_icu_prescrip_drugsfirst['SAMEDAY_ADM_TO_ICU'] = df_patient_admit_icu_prescrip_drugsfirst['DAYS_ADM_TO_ICU'].apply(lambda x: int(x<=1))

    # drop extra unwanted columns
    col_list = ['DOD','DEATHTIME']
    df_patient_admit_icu_prescrip_drugsfirst = df_patient_admit_icu_prescrip_drugsfirst.drop(col_list,axis=1)
    
    return df_patient_admit_icu_prescrip_drugsfirst
    
def pool_cycle(df_patient_admit_icu_prescrip_drugsfirst):
    """
    this function cycles through all pairs of drugs in the 
    dataset to test the relationship between pairs of drugs
    prescribed and time to icu stay, under the hypothesis
    that certain pairs of drugs may provide more information
    related to immiment icu stays compared to individual
    drugs
    """
    print('foo2')
    data_dir = 'data/physionet.org/files/mimiciii/1.4/'
    # load chart and lab events
    # look for discrepancies in duplicates, merge data and keep lab events where discrepancies exist
    chart_file = 'CHARTEVENTS.csv'
    
    # want to count lines in CHARTEVENTS to make for loop below
    row_count = 330712483 # from chart description
    
    # chart file is 35GB, so need to load it in chunks and preprocess it by left merging with existing
    skiprows = 0
    nrows = 100000  # defualt
    colnames = ['ROW_ID','SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','STORETIME','CGID','VALUE','VALUENUM','VALUEUOM','WARNING','ERROR','RESULTSTATUS','STOPPED']
    usecols = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','VALUE','VALUENUM','VALUEUOM']
    cnt=0
    
    print('foo3')
    while skiprows<=row_count:
        print('Iteration {}...'.format(cnt))
        # check if iteration ix1,ix2 started in other process,
        # skip if so
        if path.exists('{}'.format(cnt)):
            print('iteration started. skipping...')
            skiprows += nrows
            cnt += 1
            continue
        
        # if not, touch the current iteration to prevent
        # other processes from repeating
        Path('{}'.format(cnt)).touch()
        savefile = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f') + '.pickle'
            
        if skiprows + nrows > row_count:
            nrows = row_count - skiprows
        else:
            nrows = 100000
        
        df = pd.read_csv(data_dir + chart_file,sep=',', header=0, names = colnames,skiprows=skiprows, nrows=nrows, usecols=usecols)
        # convert charttime to datetime
        df.CHARTTIME = pd.to_datetime(df.CHARTTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
        
        df = df.merge(df_patient_admit_icu_prescrip_drugsfirst,how='inner',left_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],right_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
        # calculate days from chart event to icu admission
        df['DAYS_CHRT_TO_ICU'] = (df['INTIME'] - df['CHARTTIME']).dt.total_seconds()/(24*60*60)
    
        # remove rows where chart event occurred after icu admission
        icu_evnt_idx = df[df['DAYS_CHRT_TO_ICU']<=0].index
        df = df.drop(icu_evnt_idx,axis=0)
        
        with open(savefile,'wb') as to_write:
            pickle.dump(df,to_write)
        
        skiprows += nrows
        cnt += 1
    
                
def apply_async(df_patient_admit_icu_prescrip_drugsfirst):
    """runs parallel processing of pool_cycle"""
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)
    
    print('foo4')
    for n in range(num_cores):
        pool.apply_async(pool_cycle, args=(df_patient_admit_icu_prescrip_drugsfirst,))
    
    print('foo6')
    pool.close()
    pool.join()
            
if __name__ == "__main__":
    
    df_patient_admit_icu_prescrip_drugsfirst = run_chartevents_and_icustays()
    
    print('foo1')
    apply_async(df_patient_admit_icu_prescrip_drugsfirst)