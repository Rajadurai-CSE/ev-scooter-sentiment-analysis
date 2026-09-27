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
MASTER_DATA_PATH= DATA_DIR / "master_reviews.csv"
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
    # master_df_len = len(master_df)
    # df_len = len(df)
    epsilon = 1e-3
    log.info("Brand {brand}".format(brand = df['brand'].iloc[0]))
    for i in unique_df_ele:
        expected =(master_df[column] == i).mean()
        actual =(df[column] == i).mean()
        # log.info("Category {category}".format(category = i))
        # log.info("Expected {expected}".format(expected = expected))
        # log.info("Actual {actual}".format(actual = actual))
        expected = max(expected,epsilon)
        actual = max(actual,epsilon)
        ##PSI
        psi+=(actual - expected) * np.log(actual/expected)

    return psi

    

#1. Data Drift
def detect_data_drift(df:pd.DataFrame)->list[dict]:
    master_df = pd.read_csv(MASTER_DATA_PATH)
    log.info("Started Data Drift Check ...")

    drift_statistics = []

    for key,value in BRAND_MAP.items():
        master_subset = master_df[(master_df['brand'] == key)&(master_df['model']==value)]
        subset = df[(df['brand'] == key)&(df['model']==value)]
        #detect drift based on source

        bikewale_src = subset[subset['source'] == 'bikewale']
        # _91wheels_src = subset[subset['source'] == '91wheels']
        # bikedekho_src = subset[subset['source'] == 'bikedekho']

        psi = calculate_psi(master_subset[master_subset['source'] == 'bikewale'],bikewale_src,'rating')
        if psi>=0.25:
            log.info("Significant Drift Detected by PSI Test - source: Bikewale - brand: {brand} model: {model} psi-score: {psi}".format(brand = key,model = value,psi=psi))
        else:
            log.info("Data Drift is not detected - source : Bikewale - brand: {brand} model: {model} psi-score: {psi}".format(brand = key,model = value,psi=psi))

        drift_statistics.append({
            "brand":key,
            "model":value,
            "source":"bikewale",
            "psi_score":psi})

        res = ks_2samp(master_subset[(master_subset['source'] == 'bikedekho') | (master_subset['source'] == '91wheels')]['rating'],subset[(subset['source'] == 'bikedekho') | (subset['source'] == '91wheels')]['rating'])
        if res.pvalue<=0.05:
            log.info("Significant Drift Detected by KS Test - source: BikeDekho and 91wheels - brand: {brand} model: {model} ks-score: {ks} pvalue: {pvalue}".format(brand = key,model = value,ks=res.statistic, pvalue = res.pvalue))
        else:
            log.info("Data Drift is not detected - source: BikeDekho and 91wheels -  brand: {brand} model: {model} ks-score: {ks} pvalue: {pvalue}".format(brand = key,model = value,ks = res.statistic,pvalue = res.pvalue))

        drift_statistics.append({
            "brand":key,
            "model":value,
            "source":"91wheels/bikedekho",
            "k_statistic":res.statistic,
            "p_value": res.pvalue})

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

    
    









    

