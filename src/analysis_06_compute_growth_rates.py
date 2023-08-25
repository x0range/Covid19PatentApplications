import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import gzip
import pickle
import glob
import os
import scipy
import scipy.stats
import pdb

import analysis_helpers
import basic_analysis_all_subclasses

from settings import DATA_DIR, OUTPUT_DIR

def shapiro_wilk_test(df, output_directory=OUTPUT_DIR, maximum_age=None):
    """
    Function for computing Shapiro-Wilk normality test for all growth rate columns.
    @param df (pd.DataFrame): Data.
    @param output_directory: Path where computed data is saved,
    @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
    """
    print("\nCommencing Shapiro-Wilk tests:")
    growth_columns = [cn for cn in df.columns if (cn[:7]=="Growth_" and len(cn)==11)]
    sr_dfs = []
    for cn in growth_columns:
        year = int(cn[7:])
        sr = scipy.stats.shapiro(df[cn].dropna())
        sr_df = pd.DataFrame({
            "year": [year],
            "Shapiro-Wilk statistic": [sr[0]],
            "pvalue": [sr[1]]
            })
        sr_dfs.append(sr_df)
        print(year, sr)
    pdb.set_trace()
    shapiro_wilk_test_df = pd.concat(sr_dfs)
    shapiro_wilk_test_df.to_pickle(os.path.join(output_directory, "PA_growth_rates_all_subclasses_m_age" + \
                    str(maximum_age) + "_shapiro_wilk_tests.pkl.gz"), compression="gzip")
    

def compute_annual_growth_rates(start_date,
         end_date,
         do_reload=False,
         output_directory=OUTPUT_DIR, 
         data_directory=DATA_DIR,
         maximum_age=None,
         filename_addition="",
         ): 
    """Function for computing annual growth rates
        @param start_date (pandas DateTime): First date from which to the next year to compute a 
                                             growth rate. Should be April 1.
        @param end_date (pandas DateTime): Last date to which from the year before to compute a 
                                            growth rate. Should be April 1.
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param output_directory: Path where plots and computed data are saved
        @param data_directory: Path where count data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param filename_addition (str) - string to be inserted in output filenames for those to be 
                                         distinguishable
    """
    
    """ Define individual year time periods"""
    period_dict = basic_analysis_all_subclasses.multi_year_period_lists(period_start=start_date, 
                                              period_end=end_date)

    """ Obtain data"""
    period_names = {k: k for k in period_dict.keys()}
    df_counts, _ = basic_analysis_all_subclasses.count_ranges_by_code(periods = period_dict, 
                                        period_names=period_names, 
                                        do_reload=do_reload,
                                        maximum_age=maximum_age,
                                        base_data_filename="df_Covid-19_all_terms_subclasses.pkl.gz")
    
    #pdb.set_trace()
    """ Compute average growth rates"""
    #period_keys = key_lists[key]
    #df_counts["Growth_" + key] = 0
    for pkey in period_dict.keys():
        """ Catch partial years and throw error since this is not implemented"""
        assert ((df_counts[pkey + "_Length"] >= 364) | pd.isna(df_counts[pkey + "_Length"])).all(), "Partial year " + pkey
        if "Count_Old" in df_counts:
            df_counts["Growth_" + pkey] = (df_counts[pkey + "_Count"] - df_counts["Count_Old"]) /\
                                                      df_counts["Count_Old"]
        df_counts["Count_Old"] = df_counts[pkey + "_Count"]
    #df_counts["Growth_" + pkey] = df_counts["Growth_" + pkey] / (len(period_dict) - 1)
    del df_counts["Count_Old"]

    df_counts.to_pickle(os.path.join(output_directory, "PA_growth_rates_all_subclasses_m_age" + \
                    str(maximum_age) + ".pkl.gz"), compression="gzip")
    shapiro_wilk_test(df_counts)
    #pdb.set_trace()
    

if __name__ == '__main__':
    compute_annual_growth_rates(start_date=pd.to_datetime("2014-04-01"), 
                                end_date=pd.to_datetime("2021-04-01"), 
                                maximum_age=548)

"""
Commencing Shapiro-Wilk tests:
2015 ShapiroResult(statistic=0.47313815355300903, pvalue=4.015950440342278e-39)
2016 ShapiroResult(statistic=0.20146197080612183, pvalue=4.203895392974451e-45)
2017 ShapiroResult(statistic=0.052043914794921875, pvalue=0.0)
2018 ShapiroResult(statistic=0.0289497971534729, pvalue=0.0)
2019 ShapiroResult(statistic=0.0814669132232666, pvalue=0.0)
2020 ShapiroResult(statistic=0.4402048587799072, pvalue=1.5007486163379493e-40)
"""
