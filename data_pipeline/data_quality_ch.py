#Drift Quality Check Logic
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from log_manager.logger import get_logger
from scipy.stats import ks_2samp
log = get_logger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
DRIFT_STATISTICS = DATA_DIR/"drift_statistics.csv"

sources = ["91wheels","bikewale","bikedekho"]
min_rating = 0
max_rating = 5

BRAND_MAP = {
       "TVS" :   "TVS iQube",
        "Ather": "Ather Rizta",
       "Ola Electric": "Ola S1X" ,
        "Bajaj" : "Bajaj Chetak",
        "Hero": "Hero Vida V2",
    }

def calculate_psi(master_df:pd.DataFrame,df:pd.DataFrame,column:str) -> float:

    unique_df_ele = set(master_df[column].unique()).union(set(df[column].unique()))
    psi = 0
    epsilon = 1e-3
    for i in unique_df_ele:
        expected =(master_df[column] == i).mean()
        actual =(df[column] == i).mean()
        expected = max(expected,epsilon)
        actual = max(actual,epsilon)
        ##PSI
        psi+=(actual - expected) * np.log(actual/expected)

    return psi

    

#1. Data Drift
def detect_data_drift(master_df:pd.DataFrame,df:pd.DataFrame)->list[dict]:

    log.info("Started Data Drift Check ...")

    drift_statistics = []

    for key,value in BRAND_MAP.items():

        log.info("Brand : {brand} Model : {model}".format(brand=key,model=value))

        master_subset = master_df[(master_df['brand'] == key)&(master_df['model']==value)]
        subset = df[(df['brand'] == key)&(df['model']==value)]
        #detect drift based on source

        bikewale_src = subset[subset['source'] == 'bikewale']
        # _91wheels_src = subset[subset['source'] == '91wheels']
        # bikedekho_src = subset[subset['source'] == 'bikedekho']

        if len(bikewale_src)!=0:
            psi = calculate_psi(master_subset[master_subset['source'] == 'bikewale'],bikewale_src,'rating')
            if psi>=0.25:
                log.info("Significant Drift Detected by PSI Test - source: Bikewale - brand: {brand} model: {model} psi-score: {psi}".format(brand = key,model = value,psi=psi))
            else:
                log.info("Data Drift is not detected - source : Bikewale - brand: {brand} model: {model} psi-score: {psi}".format(brand = key,model = value,psi=psi))
        else:
            log.info("Empty dataframe received for psi test, skipping psi ..")
            psi = "skipped"

        drift_statistics.append({
            "brand":key,
            "model":value,
            "source":"bikewale",
            "psi_score":psi})



        if len(subset[(subset['source'] == 'bikedekho') | (subset['source'] == '91wheels')])!=0:
            res = ks_2samp(master_subset[(master_subset['source'] == 'bikedekho') | (master_subset['source'] == '91wheels')]['rating'],subset[(subset['source'] == 'bikedekho') | (subset['source'] == '91wheels')]['rating'])
            statistic = res.statistic
            pvalue = res.pvalue
            if res.pvalue<=0.05:
                log.info("Significant Drift Detected by KS Test - source: BikeDekho and 91wheels - brand: {brand} model: {model} ks-score: {ks} pvalue: {pvalue}".format(brand = key,model = value,ks=res.statistic, pvalue = res.pvalue))
            else:
                log.info("Data Drift is not detected - source: BikeDekho and 91wheels -  brand: {brand} model: {model} ks-score: {ks} pvalue: {pvalue}".format(brand = key,model = value,ks = res.statistic,pvalue = res.pvalue))
        else:
            log.info("Empty DataFrame received for ks2samp test, skipping ks2samp")
            statistic = ""
            pvalue = ""


        drift_statistics.append({
            "brand":key,
            "model":value,
            "source":"91wheels/bikedekho",
            "k_statistic":statistic,
            "p_value": pvalue})

    return drift_statistics



def feature_drift(df:pd.DataFrame) -> bool:


    # 1. Rating Drift (Based on min and max)
    min_subset_rating = df['rating'].min()
    max_subset_rating = df['rating'].max()
    feature_drifted = False

    min_rating_violated = min_rating>min_subset_rating
    max_rating_violated = max_rating<max_subset_rating

    if(min_rating_violated):
        feature_drifted = True
        log.info("Min Rating Violated")
        data_ = df[(df['rating'] == min_subset_rating)].iloc[0]
        log.info("Min Rating Violated case -  Brand: {brand} Model: {model} Rating: {rating} Index: {index} ".format(brand = data_['brand'],model = data_['model'], rating = data_['rating'],index= data_.name))
    if(max_rating_violated):  
        feature_drifted = True
        log.info("Max Rating Violated ")
        data_ = df[(df['rating'] == max_subset_rating)].iloc[0]
        log.info("Max Rating Violated case -  Brand: {brand} Model: {model} Rating: {rating} Index: {index} ".format(brand = data_['brand'],model = data_['model'], rating = data_['rating'],index= data_.name))


    #2. Check on bikewale (rating dist should be categorical not continous)
    
    bikewale_subset = df[df['source']=="bikewale"]['rating']
    subgrp = bikewale_subset[bikewale_subset != bikewale_subset.astype(int)]
    if subgrp.any():
        log.info("Rating distribution change in bikewale")
        feature_drifted = True
    return feature_drifted

    
    









    

