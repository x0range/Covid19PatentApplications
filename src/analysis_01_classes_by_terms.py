import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import gzip
import pickle
import glob
import os
import pdb

import analysis_helpers

from settings import DATA_DIR, OUTPUT_DIR


def main(do_reload=False, compute_separate=False, data_directory=DATA_DIR):


    data_filename = os.path.join(data_directory, "work", "df_Covid-19_all_terms_subclasses.pkl.gz")
    counts_filename = {
            "Covid-19": os.path.join(data_directory, "work", "df_Covid-19_all_terms_subclasses_counts.pkl.gz"),
            "Covid-19-all": os.path.join(data_directory, "work", "df_Covid-19_all_terms_subclasses_counts_fulltext.pkl.gz")}
    shares_filename = {
            "Covid-19": os.path.join(data_directory, "work", "df_Covid-19_all_terms_subclasses_shares.pkl.gz"),
            "Covid-19-all": os.path.join(data_directory, "work", "df_Covid-19_all_terms_subclasses_shares_fulltext.pkl.gz")}

    search_terms = ["Covid-19", "Covid19", "2019-nCov", "SARS-CoV-2", ["pandemic", "coronavirus"], ["pandemic", "covid"]]
    #pdb.set_trace()
    if not do_reload:
        try:
            df = pd.read_pickle(data_filename, compression="gzip")
        except:
            print("Loading has failed: ", sys.exc_info())
            do_reload = True
        # TODO: assert search_terms columns are present
    if do_reload:
        df = analysis_helpers.df_of_pa_by_classification(classification="subclass", do_reload=True, 
                                   search_terms=search_terms, 
                                   extract_terms=True,
                                   retain_columns=['country', 'city', 'state', 'publication_date'])
        #del df['ID']
        #pdb.set_trace()
        df["age_at_publication"] = df.apply(lambda row: (pd.to_datetime(row['publication_date']) - pd.to_datetime(row['Date'])).days, axis=1)
        df.to_pickle(data_filename, compression="gzip")
        
    """ Join search elements (title, abstract, claims) - plus description ?"""
    df["Covid-19"] = False
    df["Covid-19-all"] = False
    for st in search_terms:
        if isinstance(st, list):
            st = "_".join(st)
        #df["Covid-19"] = df["Covid-19"] | df["title_contains_" + st] | df["abstract_contains_" + st] | df["claims_contains_" + st] 
        #df["Covid-19-all"] = df["Covid-19-all"] | df["description_contains_" + st] 
        df["Covid-19_"+st], df["Covid-19-all_"+st] = merge(df, st)
        df["Covid-19"] = df["Covid-19_"+st] | df["Covid-19"] 
        df["Covid-19-all"] = df["Covid-19-all_"+st] | df["Covid-19-all"] 
    
    for fulltext in [False, True]:
        compute(df=df, st="", fulltext=fulltext, show=True)
    if compute_separate:
        for st in search_terms:
            if isinstance(st, list):
                st = "_".join(st)
            for fulltext in [False, True]:
                compute(df=df, st=st, fulltext=fulltext, show=True)
        
    #for icolumn in ["Covid-19", "Covid-19-all"]:
    #    df2 = df[df[icolumn]==True]
    #    df2 = df2[["Classification", icolumn]]
    #    df_counts = df2.groupby("Classification").agg("count").reset_index()
    #    df_counts.columns = ["Classification", "Count"]
    #    df_counts = df_counts.sort_values('Count', ascending=False)
    #    df_counts.to_pickle(counts_filename[icolumn], compression="gzip")
    #    print(df_counts.iloc[:25])
    #    
    #    """ Convert to int"""
    #    df[icolumn] = df[icolumn] * 1.0
    #    #print(df[icolumn])
    #    
    #    df2 = df[["Classification", icolumn]]
    #    df_shares = df2.groupby("Classification").agg("mean").reset_index()
    #    df_shares.columns = ["Classification", "Share"]
    #    df_shares = df_shares.sort_values('Share', ascending=False)
    #    df_shares.to_pickle(shares_filename[icolumn], compression="gzip")
    #    print(df_shares.iloc[:25])

def merge(df, st):
    """
    Function for merging columns of documents identified to contain search term in different fields
    @param df (DataFrame)
    @param st (str) search term column name
    @return (df, df) series with and without description field identification included
    """
    series = df["title_contains_" + st] | df["abstract_contains_" + st] | df["claims_contains_" + st] 
    series_all = series | df["description_contains_" + st] 
    return series, series_all
    
    
def compute(df, st, fulltext=False, show=False, reduceSample=True, data_directory=DATA_DIR):
    """
    Function to compute number and share of of occurrences per subclass for each search term (or all together)
    @param df (DataFrame)
    @param st (str) search term column name
    @param fulltext (bool) include description field or not
    @param show (bool) also print results in terminal (or just save them)
    @param reduceSample (bool) reduce the data to those patent applications used in the study (filed after 
           March 2014, until March 2021, published within 555 days)
    @param data_directory (str) path to data directory
    """
    
    """ Local variables"""
    rSmaximum_age = 555
    rSstart = pd.to_datetime("2014-04-01")
    rSend = pd.to_datetime("2021-03-31")
    
    """ Print status message"""
    print("\n\n\n---------------------")
    if fulltext:
        print("Identified term {0:s} this often in fulltext (incl.description):".format(st))
    else:
        print("Identified term {0:s} this often in title, abstract, claims:".format(st))
    print("---------------------\n\n\n")
    
    """Prepare column name and file name strings"""
    sep = "_" if (len(st) > 0) else ""
    rs = "_reduced" if reduceSample else ""
    if fulltext:
        icolumn = "Covid-19-all" + sep + st
        counts_filename = os.path.join(data_directory, "work", "df_Covid-19_all" + sep + st + "_terms_subclasses_counts_fulltext_weighted" + rs + ".pkl.gz")
        shares_filename = os.path.join(data_directory, "work", "df_Covid-19_all" + sep + st + "_terms_subclasses_shares_fulltext_weighted" + rs + ".pkl.gz")
    else:
        icolumn = "Covid-19" + sep + st
        counts_filename = os.path.join(data_directory, "work", "df_Covid-19" + sep + st + "_terms_subclasses_counts_fulltext_weighted" + rs + ".pkl.gz")
        shares_filename = os.path.join(data_directory, "work", "df_Covid-19" + sep + st + "_terms_subclasses_shares_fulltext_weighted" + rs + ".pkl.gz")
    
    """Reduce sample"""
    if reduceSample:
        df = df[(df["age_at_publication"]<=rSmaximum_age) & (df["Date"]>=rSstart) & (df["Date"]<=rSend)]
    
    """ Extract rows with matches"""
    df2 = df[df[icolumn]==True]
    df2 = df2[["Classification", "Classification_share"]]

    """ Compute counts"""
    #df_counts = df2.groupby("Classification").agg("count").reset_index()
    df_counts = df2.groupby("Classification").agg("sum").reset_index()
    pdb.set_trace()
    df_counts.columns = ["Classification", "Count"]
    df_counts = df_counts.sort_values('Count', ascending=False)
    df_counts.to_pickle(counts_filename, compression="gzip")
    if show:
        print(df_counts.iloc[:25])
    
    """ Convert to int"""
    df[icolumn] = df[icolumn] * 1.0
    #print(df[icolumn])
    
    df["Classification_share_filtered"] = df[icolumn] * df["Classification_share"]
    #df2 = df[["Classification", icolumn]]
    df2 = df[["Classification", "Classification_share_filtered"]]
    del df["Classification_share_filtered"]
    #pdb.set_trace()
    df_shares = df2.groupby("Classification").agg("mean").reset_index()
    df_shares.columns = ["Classification", "Share"]
    df_shares = df_shares.sort_values('Share', ascending=False)
    df_shares.to_pickle(shares_filename, compression="gzip")
    if show:
        print(df_shares.iloc[:25])


            
    
if __name__ == '__main__':
    main(do_reload=False, compute_separate=True)
