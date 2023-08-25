import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
import statsmodels.robust.robust_linear_model as roblm
import numpy as np
import seaborn
import os
from settings import DATA_DIR, OUTPUT_DIR
import pdb

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


def residuals_distplot(residuals, filename):
    """
    @param residuals (list-like): residuals
    @param filename (str): name of pdf file to be created
    """
    plt.figure()
    seaborn.distplot(residuals)
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.savefig(filename)
    
    plt.figure()
    seaborn.distplot(residuals)
    plt.xlabel("Residuals")
    plt.yscale("log")
    plt.ylim(10**(-8), 10**(+1))
    plt.tight_layout()
    plt.savefig(filename[:-4] + "_log.pdf")
    
    


df = pd.read_pickle(os.path.join(OUTPUT_DIR, "PA_growth_rates_all_subclasses_m_age548.pkl.gz"), compression='gzip')
#df_cov = pd.read_pickle("../data/work/df_Covid-19_subclasses_shares.pkl.gz", compression='gzip') 
df_cov = pd.read_pickle(os.path.join(DATA_DIR, "work", "df_Covid-19_terms_subclasses_shares_fulltext_weighted_reduced.pkl.gz"), compression='gzip') 
#df_cov_upstream = pd.read_pickle("../data/work/df_upstream_from_Covid-19_abs_43.pkl.gz", compression='gzip') 
df_cov_upstream = pd.read_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_shares.pkl.gz"), compression='gzip') 
df = pd.merge(df, df_cov, on="Classification", how='left')
df["Covid_mentioned"] = df["Share"]
del df["Share"]
df_cov_upstream["Upstream_from_Covid_mentioned"] = df_cov_upstream["Indirect_Covid_relevance"]
df = pd.merge(df, df_cov_upstream, on="Classification", how='left')

growth_columns = [cn for cn in df.columns if (cn[:7]=="Growth_" and len(cn)==11)]
growth_years = [int(cn[7:]) for cn in growth_columns]
growth_columns_pre_covid = [cn for cn in growth_columns if int(cn[7:])<2020]
growth_columns_covid = [cn for cn in growth_columns if int(cn[7:])>=2020]
df["MaxGrowth"] = df[growth_columns_pre_covid].max(axis=1) 
df["MinGrowth"] = df[growth_columns_pre_covid].min(axis=1) 
df["AvgGrowth"] = df[growth_columns_pre_covid].mean(axis=1) 
df["MedGrowth"] = df[growth_columns_pre_covid].median(axis=1) 

count_columns =  [cn for cn in df.columns if (cn[4:10]=="_Count" and len(cn)==10)]
count_columns_pre_covid =  [cn for cn in count_columns if int(cn[:4])<2020]
df["Sum"] = df[count_columns].sum(axis=1) 
df["Sum_pre_covid"] = df[count_columns_pre_covid].sum(axis=1) 
df["logSum_pre_covid"] = df["Sum_pre_covid"].apply(np.log) 

df["Mainsection"] = df['Classification'].str[0]
df["col"] = df["Mainsection"].map({"A": "#e6194b", 
                                       "B": "#3cb44b",
                                       "C": "#4363d8", 
                                       "D": "#f58231",
                                       "E": "#911eb4", 
                                       "F": "#42d4f4",
                                       "G": "#f032e6", 
                                       "H": "#bfef45",
                                       "Y": "#000075"})

# sort by column
df.sort_values(by=["Mainsection", "MedGrowth"], ascending=True, inplace=True)
#df.reset_index(drop=True, inplace=True)

df["relative_g_2020"] = (-1) * (df["Growth_2020"] - df["MinGrowth"]) / (df["MaxGrowth"] - df["MinGrowth"])
df.sort_values(by=["Mainsection","relative_g_2020"], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
#pdb.set_trace()
print(df)

with open(os.path.join(OUTPUT_DIR, "regressions_continuous.txt"), "w") as wfile:
    wfile.write("")

models = {'m1': "Growth_2020 ~ Covid_mentioned", # m0
          'm2': "Growth_2020 ~ logSum_pre_covid + Covid_mentioned", #m1
          'm3': "Growth_2020 ~ logSum_pre_covid + Mainsection + Covid_mentioned", #m1a
          'm4': "Growth_2020 ~ logSum_pre_covid + Covid_mentioned + Upstream_from_Covid_mentioned", #m1b
            #"Growth_2020 ~ logSum_pre_covid + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m1c
          'm5a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Covid_mentioned", #m2
          'm5g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Covid_mentioned", #m2a
          'm6a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned", #m3
          'm6g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Mainsection + Covid_mentioned", #m3a
            #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned",
            #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned",
          'm7a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m4
          'm7g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m4a
            #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Covid_mentioned + Upstream_from_Covid_mentioned",#m5
            #"Growth_2020 ~ logSum_pre_covid + Growth_2019 + Covid_mentioned + Upstream_from_Covid_mentioned"#m5a
            }
            
fits_all = {}

for key in models.keys():
    model = models[key]
    smodel = sm.ols(formula=model, data=df)
    fit = smodel.fit()
    fits_all[key] = fit
    print("\n", model)
    print(fit.summary())
    with open(os.path.join(OUTPUT_DIR, "regressions_continuous.txt"), "a") as wfile:
        wfile.write("\n\nOLS\n\n")
        wfile.write("\n" + str(model))
        wfile.write("\n" + fit.summary().as_text())
    if model == "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned":
        residuals_distplot(residuals = fit.resid, filename="residuals_distplot_continuous.pdf")

output_lists = {'Avg6': ['m1', 'm2', 'm3', 'm4', 'm5a', 'm6a'],
             'Avg7': ['m1', 'm2', 'm3', 'm4', 'm5a', 'm6a', 'm7a'],
             'Gro6': ['m1', 'm2', 'm3', 'm4', 'm5g', 'm6g'],
             'Gro7': ['m1', 'm2', 'm3', 'm4', 'm5g', 'm6g', 'm7g']}
headline_string = {'Avg6': "\n\nOLS - Summary Table AvgGrowth (6) \n\n",
                   'Avg7': "\n\nOLS - Summary Table AvgGrowth (7) \n\n",
                   'Gro6': "\n\nOLS - Summary Table Growth 2019 (6) \n\n",
                   'Gro7': "\n\nOLS - Summary Table Growth 2019 (7) \n\n"}

for key_ol in output_lists.keys():
    output_list = output_lists[key_ol]
    out_table = summary_col([fits_all[key] for key in output_list], stars=True)
    print(out_table)
    with open(os.path.join(OUTPUT_DIR, "regressions_continuous.txt"), "a") as wfile:
        wfile.write(headline_string[key_ol])
        wfile.write("\n" + out_table.as_text())

"""

"""


corr_columns = ["Growth_2019", "Growth_2020", "AvgGrowth", "logSum_pre_covid", "Covid_mentioned", "Upstream_from_Covid_mentioned"]
df_corr = df[corr_columns]
print("\n\n")
print(df_corr.corr())

"""

"""

robmodels = {'m1': "Growth_2020 ~ Covid_mentioned", # m0
             'm2': "Growth_2020 ~ logSum_pre_covid + Covid_mentioned", #m1
             'm3': "Growth_2020 ~ logSum_pre_covid + Mainsection + Covid_mentioned", #m1a
             'm4': "Growth_2020 ~ logSum_pre_covid + Covid_mentioned + Upstream_from_Covid_mentioned", #m1b
             #"Growth_2020 ~ logSum_pre_covid + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m1c
             'm5a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Covid_mentioned", #m2
             'm5g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Covid_mentioned", #m2a
             'm6a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned", #m3
             'm6g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Mainsection + Covid_mentioned", #m3a
             #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned",
             #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned",
             'm7a': "Growth_2020 ~ logSum_pre_covid + AvgGrowth + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m4
             'm7g': "Growth_2020 ~ logSum_pre_covid + Growth_2019 + Mainsection + Covid_mentioned + Upstream_from_Covid_mentioned", #m4a
             #"Growth_2020 ~ logSum_pre_covid + AvgGrowth + Covid_mentioned + Upstream_from_Covid_mentioned",#m5
             #"Growth_2020 ~ logSum_pre_covid + Growth_2019 + Covid_mentioned + Upstream_from_Covid_mentioned"#m5a
            }

fits_all = {}

print("\n\n")
for key in robmodels.keys():
    model = robmodels[key]
    smodel = roblm.RLM.from_formula(formula=model, data=df)
    fit = smodel.fit()
    fits_all[key] = fit
    print("\n", model)
    print(fit.summary())
    with open(os.path.join(OUTPUT_DIR, "regressions_continuous.txt"), "a") as wfile:
        wfile.write("\n\nRobust - Huber T norm\n\n")
        wfile.write("\n" + str(model))
        wfile.write("\n" + fit.summary().as_text())

output_lists = {'Avg6': ['m1', 'm2', 'm3', 'm4', 'm5a', 'm6a'],
             'Avg7': ['m1', 'm2', 'm3', 'm4', 'm5a', 'm6a', 'm7a'],
             'Gro6': ['m1', 'm2', 'm3', 'm4', 'm5g', 'm6g'],
             'Gro7': ['m1', 'm2', 'm3', 'm4', 'm5g', 'm6g', 'm7g']}
headline_string = {'Avg6': "\n\nRobust - Huber T norm - Summary Table AvgGrowth (6) \n\n",
                   'Avg7': "\n\nRobust - Huber T norm - Summary Table AvgGrowth (7) \n\n",
                   'Gro6': "\n\nRobust - Huber T norm - Summary Table Growth 2019 (6) \n\n",
                   'Gro7': "\n\nRobust - Huber T norm - Summary Table Growth 2019 (7) \n\n"}

for key_ol in output_lists.keys():
    output_list = output_lists[key_ol]
    out_table = summary_col([fits_all[key] for key in output_list], stars=True)
    print(out_table)
    with open(os.path.join(OUTPUT_DIR, "regressions_continuous.txt"), "a") as wfile:
        wfile.write(headline_string[key_ol])
        wfile.write("\n" + out_table.as_text())

"""

"""
