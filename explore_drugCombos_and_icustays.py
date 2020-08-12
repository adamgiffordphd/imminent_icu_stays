#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import multiprocessing as mp
from pathlib import Path
from os import path, remove
import pickle
import datetime
"""
note: this represents 90% of code used, rest is part of
jupyter notebook "assets"
"""
def run_drugCombos_and_icustays():
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
    data_dir = 'data/physionet.org/files/mimiciii/1.4/'
    patient_file = 'PATIENTS.csv'
    df_patients = pd.read_csv(data_dir + patient_file)
    
    df_patients=df_patients.drop(['ROW_ID','DOD_HOSP','DOD_SSN'],axis=1)
    
    # convert date strings to datetime
    df_patients.DOB = pd.to_datetime(df_patients.DOB,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # import admissions info
    admissions_file = 'ADMISSIONS.csv'
    df_admissions = pd.read_csv(data_dir + admissions_file)
    df_admissions = df_admissions.drop(['ROW_ID','RELIGION','LANGUAGE','MARITAL_STATUS','ETHNICITY'],axis=1)

    # convert time strings to datetime
    df_admissions.ADMITTIME = pd.to_datetime(df_admissions.ADMITTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_admissions.DISCHTIME = pd.to_datetime(df_admissions.DISCHTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    # calculate length of hospital stay
    df_admissions['HOSPITAL_DAYS'] = (df_admissions['DISCHTIME'] - df_admissions['ADMITTIME']).dt.total_seconds()/(24*60*60)
    # negative admit days = dead on arrival, remove
    doa_idx = df_admissions[df_admissions['HOSPITAL_DAYS']<0].index
    df_admissions = df_admissions.drop(doa_idx,axis=0)
    
    # merge patient and admissions df
    df_patient_admit = df_patients.merge(df_admissions,how='left',left_on=['SUBJECT_ID'],right_on=['SUBJECT_ID'])
    
    # calculate age at admit
    df_patient_admit['ADMIT_AGE'] = df_patient_admit['ADMITTIME'].dt.year - df_patient_admit['DOB'].dt.year
    
    # 2. Remove patients <X
    age = 10
    child_idx = df_patient_admit[df_patient_admit['ADMIT_AGE']<age].index
    child_patients = df_patient_admit.iloc[child_idx]['SUBJECT_ID'].unique()
    df_patient_admit = df_patient_admit.drop(child_idx, axis=0)
    
    # 3. Load icustays
    icustays_file = 'ICUSTAYS.csv'
    df_icustays = pd.read_csv(data_dir + icustays_file)
    
    child_idx = df_icustays[df_icustays['SUBJECT_ID'].isin(child_patients)].index
    df_icustays = df_icustays.drop(child_idx,axis=0)
    
    df_icustays = df_icustays.drop(['ROW_ID'],axis=1)
    df_icustays.INTIME = pd.to_datetime(df_icustays.INTIME,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    #join admissions and icu stays
    df_admit_icu = df_admissions.merge(df_icustays,how='left',left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'])
    
    cols_to_rmv = list(set(df_admit_icu.columns) & set(df_patient_admit))
    # keep SUBJECT_ID and HADM_ID for merge
    sbj_idx = cols_to_rmv.index('SUBJECT_ID')
    cols_to_rmv.pop(sbj_idx)
    h_idx = cols_to_rmv.index('HADM_ID')
    cols_to_rmv.pop(h_idx)
    
    df_admit_icu = df_admit_icu.drop(cols_to_rmv,axis=1)
    
    df_patient_admit_icu = df_patient_admit.merge(df_admit_icu,how='left',left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'])
    
    df_patient_admit_icu['DAYS_ADM_TO_ICU'] = (df_patient_admit_icu['INTIME'] - df_patient_admit_icu['ADMITTIME']).dt.total_seconds()/(24*60*60)
    
    # 4. Load prescriptions
    prescrips_file = 'PRESCRIPTIONS.csv'
    df_prescrips = pd.read_csv(data_dir + prescrips_file,low_memory=False)
    
    df_prescrips = df_prescrips.drop(['ROW_ID','GSN','DRUG','DRUG_NAME_POE','DRUG_NAME_GENERIC','FORMULARY_DRUG_CD'],axis=1)
    ndc_nan_idx = df_prescrips[df_prescrips['NDC'].isna()].index
    df_prescrips=df_prescrips.drop(ndc_nan_idx,axis=0)
    
    df_prescrips['NDC'] = df_prescrips['NDC'].astype('int64')
    
    child_idx = df_prescrips[df_prescrips['SUBJECT_ID'].isin(child_patients)].index
    df_prescrips = df_prescrips.drop(child_idx,axis=0)
    
    df_prescrips.STARTDATE = pd.to_datetime(df_prescrips.STARTDATE,format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    df_patient_admit_icu_prescrip = df_patient_admit_icu.merge(df_prescrips,how='left',left_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],right_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    
    df_patient_admit_icu_prescrip['DAYS_ADM_TO_DRUG'] = (df_patient_admit_icu_prescrip['STARTDATE'] - df_patient_admit_icu_prescrip['ADMITTIME']).dt.total_seconds()/(24*60*60)
    
    df_patient_admit_icu_prescrip['DAYS_DRUG_BEFORE_ICU'] = df_patient_admit_icu_prescrip['DAYS_ADM_TO_ICU'] - df_patient_admit_icu_prescrip['DAYS_ADM_TO_DRUG']
    
    df_patient_admit_icu_prescrip_drugsfirst = df_patient_admit_icu_prescrip[df_patient_admit_icu_prescrip['DAYS_DRUG_BEFORE_ICU']>0]
    
    prescrips_by_combo_before_icu = df_patient_admit_icu_prescrip_drugsfirst.groupby(['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).agg({'NDC': ['count', list]})
    
    prescrips_by_combo_before_icu.sort_values(by=('NDC','count'),ascending=False).head(5)
    
    # find all subject, hadm, icu_stay where count ==1 and remove (since no interaction)
    prescrips_by_combo_before_icu = prescrips_by_combo_before_icu[prescrips_by_combo_before_icu[('NDC',  'count')]>1]
    
    df_keep_ix = prescrips_by_combo_before_icu.index
    df_keep_subjs = [x[0] for x in df_keep_ix]
    df_keep_hadm = [x[1] for x in df_keep_ix]
    df_keep_icustay = [x[2] for x in df_keep_ix]
    
    df_ix1 = df_patient_admit_icu_prescrip_drugsfirst['SUBJECT_ID'].apply(lambda x: x in df_keep_subjs)
    df_ix2 = df_patient_admit_icu_prescrip_drugsfirst['HADM_ID'].apply(lambda x: x in df_keep_hadm)
    df_ix3 = df_patient_admit_icu_prescrip_drugsfirst['ICUSTAY_ID'].apply(lambda x: x in df_keep_icustay)
    
    df_patient_admit_icu_prescrip_drugsfirst = df_patient_admit_icu_prescrip_drugsfirst[df_ix1 & df_ix2 & df_ix3]
    

    # get unique drugs
    drugs = df_patient_admit_icu_prescrip['NDC'].unique()
    
    drugs = [x for x in drugs if x==x and x!=0.0]
    
    return drugs,prescrips_by_combo_before_icu,df_patient_admit_icu_prescrip_drugsfirst
    
def pool_cycle(drugs,df_drug_combos,df_patient_info):
    """
    this function cycles through all pairs of drugs in the 
    dataset to test the relationship between pairs of drugs
    prescribed and time to icu stay, under the hypothesis
    that certain pairs of drugs may provide more information
    related to immiment icu stays compared to individual
    drugs
    """
    savefile = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f') + '.pickle'
    
    drug_combos = []
    mn_days_drug_b4_icu = []
    se_days_drug_b4_icu = []
    combo_total = []
    ix1s = []
    ix2s = []
    for ix1, drug1 in enumerate(drugs):
        print('Iteration {} out of {}...'.format(ix1,len(drugs)))
        for ix2 in range(ix1+1,len(drugs)):
            # check if iteration ix1,ix2 started in other process,
            # skip if so
            if path.exists('{}_{}'.format(ix1,ix2)):
                print('iteration started. skipping...')
                continue
            
            # if not, touch the current iteration to prevent
            # other processes from repeating
            Path('{}_{}'.format(ix1,ix2)).touch()
            if ix2 % 500 == 0:
                print('     Subiteration {} out of {}...'.format(ix2,len(drugs)))
                
            drug2 = drugs[ix2]
            drug_combos.append((drug1,drug2))
            
            # find rows where both drugs in NDC list
            drg1_in = df_drug_combos[('NDC',  'list')].apply(lambda x: drug1 in x)
            drg2_in = df_drug_combos[('NDC',  'list')].apply(lambda x: drug2 in x)
            both_ix = df_drug_combos.index[drg1_in & drg2_in]
            
            both_subjs = [x[0] for x in both_ix]
            both_hadm = [x[1] for x in both_ix]
            both_icustay = [x[2] for x in both_ix]
            
            df_ix1 = df_patient_info['SUBJECT_ID'].apply(lambda x: x in both_subjs)
            df_ix2 = df_patient_info['HADM_ID'].apply(lambda x: x in both_hadm)
            df_ix3 = df_patient_info['ICUSTAY_ID'].apply(lambda x: x in both_icustay)
            
            # how many times that combination occurred
            combo_total.append(sum(df_ix1 & df_ix2 & df_ix3))
            
            # mean/sem days drug prescribed prior to icu stay
            mn_days_drug_b4_icu.append(df_patient_info[df_ix1 & df_ix2 & df_ix3].DAYS_DRUG_BEFORE_ICU.mean())
            se_days_drug_b4_icu.append(df_patient_info[df_ix1 & df_ix2 & df_ix3].DAYS_DRUG_BEFORE_ICU.sem())
            # keep ix vals for combining parallel processes later
            ix1s.append(ix1)
            ix2s.append(ix2)
            
            # save data
            data_to_save = (combo_total, mn_days_drug_b4_icu, 
                            se_days_drug_b4_icu,ix1s,ix2s)
            with open(savefile,'wb') as to_write:
                pickle.dump(data_to_save,to_write)
                
            # delete touch file when complete
            remove('{}_{}'.format(ix1,ix2))
                
def apply_async(drugs,df_drug_combos,df_patient_info):
    """runs parallel processing of pool_cycle"""
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)
    args = (drugs,df_drug_combos,df_patient_info)
    
    for n in range(num_cores):
        pool.apply_async(pool_cycle, args=args)
        
    pool.close()
    pool.join()
            
if __name__ == "__main__":
    
    drugs, df_drug_combos,df_patient_info = run_drugCombos_and_icustays()
    
    apply_async(drugs,df_drug_combos,df_patient_info)