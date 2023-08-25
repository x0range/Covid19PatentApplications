import pandas as pd
import pickle
import gzip
import os
import numpy as np
import scipy
import scipy.stats
#import pdb # pdb and pd.pivot conflicts for python 3.10.2 and pandas 1.4.1
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)



from settings import DATA_DIR, OUTPUT_DIR

def bootstrap_confidence(start_value, growth_mean, growth_std, future_length, confidence_levels, number_replications=1000000, actual_next_value=None):
    """
    Function for bootstrapping confidence intervals of an exponential growth process
    @param start_value (numeric): Value of the process at the start of the prediction
    @param growth_mean (numeric): Gaussian fit mu of the growth
    @param growth_std (numeric): Gaussian fit sigma of the growth
    @param future_length (int): How many periods to predict
    @param confidence_levels (list of float): List of confidence levels to return
    @param number_replications (int): Sample size for bootstrapping
    @param actual_next_value (float or None): Actual next value of growth process to determine it's quantile
    @return (pd.DataFrame, float): 
            upper and lower value for each confidence level; 
            quantile of actual growth
    """
    array_list = [np.ones(number_replications) * start_value]
    for i in range(future_length):
        growth_rate_rvs = scipy.stats.norm.rvs(loc=growth_mean, scale=growth_std, size=number_replications)
        array_list.append(array_list[-1] * (1+growth_rate_rvs))
    df = pd.DataFrame()
    for cl in confidence_levels:
        col_name_lower = "{0:.2f}".format(cl)[2:4] + "_lower"
        col_name_upper = "{0:.2f}".format(cl)[2:4] + "_upper"
        df[col_name_lower] = [np.quantile(sample,(1-cl)/2.) for sample in array_list]
        df[col_name_upper] = [np.quantile(sample,1-(1-cl)/2.) for sample in array_list]
    if actual_next_value is not None:
        quantile_of_actual_growth = np.sum(array_list[1] < actual_next_value) / number_replications
    return df, quantile_of_actual_growth


def plot_extrapolation(ds, mainsection, start_year, end_year):
    """
    @param ds (pandas.DataSeries) - PA count data
    @param mainsection (str of len 0 or 1) - letter of mainsection to be considered 
                                             (if empty string, include all)
    @param start_year (int) First year of range
    @param end_year (int) Last year of range
    """
    #ds = np.asarray([292147, 291285, 289372, 295013, 295023, 313998, 305029])
    ds = np.asarray(ds)
    years = np.arange(start_year, end_year + 1)
    growth = (ds[1:] - ds[:-1]) / ds[:-1]
    norm_mu, norm_sigma = scipy.stats.norm.fit(growth)
    future_length = 2
    future = {}
    idx_2019 = np.argwhere(years==2019)[0][0]

    future['mean'] = [ds[idx_2019] * (1 + norm_mu)**i for i in range(future_length + 1)]
    future['time'] = np.arange(2019, 2019 + 1 + future_length)
    future_df, true_quantile = bootstrap_confidence(start_value=ds[idx_2019], 
                                                    growth_mean=norm_mu, 
                                                    growth_std=norm_sigma, 
                                                    future_length=future_length, 
                                                    confidence_levels=[0.5, 0.9],
                                                    actual_next_value=ds[idx_2019+1])
    print("\nTrue quantile of {0:d}: {1:f}".format(ds[idx_2019+1], true_quantile))

    colors = {"A": "#e6194b", 
              "B": "#3cb44b",
              "C": "#4363d8", 
              "D": "#f58231",
              "E": "#911eb4", 
              "F": "#42d4f4",
              "G": "#f032e6", 
              "H": "#bfef45",
              "Y": "#000075",
              "": '#0000ff'}

    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax[0][0].plot(years, ds, color="k", label="Empirical data")
    ax[0][0].fill_between(future['time'], future_df['50_lower'], future_df['50_upper'], color=colors[mainsection], alpha=0.35, label="50% quantile")
    ax[0][0].fill_between(future['time'], future_df['90_lower'], future_df['90_upper'], color=colors[mainsection], alpha=0.15, label="90% quantile")
    ax[0][0].plot(future['time'], future['mean'], color=colors[mainsection], label="Extrapolation")
    ax[0][0].set_ylabel("# Patent applications")
    ax[0][0].legend(loc='upper left')
    plt.tight_layout()
    pdffilename = "numbers_PA_all_" + mainsection + ".pdf" if len(mainsection)>0 else "numbers_PA_all.pdf"
    plt.savefig(pdffilename)


df = pd.read_pickle(os.path.join(DATA_DIR, "work", "df_patents_list_short_None.pkl.gz"), compression="gzip")

df["pd"]=pd.to_datetime(df['publication_date'])
df["ad"]=pd.to_datetime(df['application_date'])
df["age_at_publication"] = df["pd"]-df["ad"]
del df["pd"]
del df["ad"]
df["age_at_publication"] = df["age_at_publication"].dt.days
#df["age_at_publication"] = df.apply(lambda row: (pd.to_datetime(row['publication_date']) - pd.to_datetime(row['application_date'])).days, axis=1)
df["Year"] = pd.DatetimeIndex(df["application_date"]).year
df["Month"] = pd.DatetimeIndex(df["application_date"]).month
df["mainsections"] = df.apply(lambda row: "".join(cc[0] for cc in row['CPC_codes']), axis=1)

start_year, end_year_plots, end_year = 2014, 2020, 2021
assert end_year_plots <= end_year

#for maximum_age in [584, 366]:
for maximum_age in [584]:
    with open(os.path.join(OUTPUT_DIR, "numbers_" + str(maximum_age) + ".txt"), "w") as wfile:
        wfile.write("")
    df_reduced = df[df["age_at_publication"] <= maximum_age]
    #pdb.set_trace()
    
    for mainsection in ["", "A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        print("Commencing Mainsecition", mainsection)
        if len(mainsection) > 0:
            df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
            df_ms = df_reduced[df_reduced["ms_present"]]
        else:
            df_ms = df_reduced.copy()
        df_counts = df_ms[["Year", "Month", "application_date"]].groupby(["Year", "Month"]).agg("count").reset_index()
        df_counts.columns = ["Year", "Month", "Count"]
        df_counts = df_counts[df_counts["Year"]>=start_year]
        df_counts = df_counts[df_counts["Year"]<=end_year]
        df_counts2 = df_counts.pivot(index="Year", columns="Month", values="Count")
        print(df_counts2)
        print(df_counts.groupby("Year").agg("sum"))
        print(np.sum(df_counts.groupby("Year").agg("sum")))
        with open(os.path.join(OUTPUT_DIR, "numbers_" + str(maximum_age) + ".txt"), "a") as wfile:
            if mainsection == "":
                wfile.write("\n\nAll patent applications \n")
            else:
                wfile.write("\n\nMainsection "+ str(mainsection) + "\n")
            wfile.write("\n"+ str(df_counts2))
            wfile.write("\n"+ str(df_counts.groupby("Year").agg("sum")))
            wfile.write("\n"+ str(np.sum(df_counts.groupby("Year").agg("sum"))))
        
        df_counts = df_counts[df_counts["Year"]<=end_year_plots]
        print(np.sum(df_counts.groupby("Year").agg("sum")))
        ds = df_counts.groupby("Year").agg("sum")["Count"]
        plot_extrapolation(ds, mainsection, start_year, end_year_plots)
    

"""
Commencing Mainsecition 
Month     1      2      3      4      5      6      7      8      9      10     11     12
Year                                                                                     
2014   21904  21609  32167  22483  23135  24611  23823  22481  25933  24505  21226  28270
2015   20423  20810  26877  23128  22550  26641  24536  23144  27076  24477  23112  28511
2016   20522  22782  27275  22579  23174  26498  21899  25876  25791  22662  23250  27064
2017   22159  21881  27545  21900  24706  26506  22297  24943  25322  24090  25587  28077
2018   23192  20467  26131  22091  24491  24493  23651  26030  24664  26469  25784  27560
2019   24298  22332  27325  25903  25786  25603  26241  26371  27299  27688  25805  29347
2020   24511  23450  27935  24642  23470  26408  25463  23503  28847  23961  23305  29534
      Month   Count
Year               
2014     78  292147
2015     78  291285
2016     78  289372
2017     78  295013
2018     78  295023
2019     78  313998
2020     78  305029

True quantile of 305029: 0.105939
findfont: Font family ['normal'] not found. Falling back to DejaVu Sans.
Commencing Mainsecition A
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month    1     2     3     4     5     6     7     8     9     10    11    12
Year                                                                         
2014   3523  3606  5826  3754  3938  3845  3879  3681  3959  4158  3540  4314
2015   3465  3747  4571  4200  4103  4474  4387  3840  4771  4444  4018  4701
2016   3758  4150  4707  4521  4543  5008  4185  4713  4828  4542  4615  5026
2017   4288  4275  5286  4271  5094  5388  4498  4950  4707  4804  4786  5238
2018   4680  4131  5045  4485  5108  4825  4591  5131  4476  5332  4697  5166
2019   4669  4554  5251  5215  4979  4986  4898  5025  5039  5303  4923  5444
2020   4738  4632  5276  5070  4841  5165  5119  4725  5666  5259  4842  5585
      Month  Count
Year              
2014     78  48023
2015     78  50721
2016     78  54596
2017     78  57585
2018     78  57667
2019     78  60286
2020     78  60918

True quantile of 60918: 0.125522
Commencing Mainsecition B
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month    1     2     3     4     5     6     7     8     9     10    11    12
Year                                                                         
2014   2565  2638  4086  2818  2886  3254  3141  3082  3489  3515  3045  3772
2015   2817  2934  3970  3388  3226  3656  3464  3309  3621  3743  3497  4052
2016   3163  3511  4252  3659  3605  4218  3754  4166  4198  3896  4259  4626
2017   3764  4008  5112  3948  4513  4746  4098  4514  4634  4601  4776  5025
2018   4384  3924  5056  4165  4497  4488  4345  4696  4342  4919  4798  4737
2019   4263  4044  5094  4616  4528  4548  4670  4685  4864  5207  4720  5267
2020   4443  4242  5088  4245  3901  4471  4306  3972  5021  4240  4098  4870
      Month  Count
Year              
2014     78  38291
2015     78  41677
2016     78  47307
2017     78  53739
2018     78  54351
2019     78  56506
2020     78  52897

True quantile of 52897: 0.043781
Commencing Mainsecition C
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month    1     2     3     4     5     6     7     8     9     10    11    12
Year                                                                         
2014   1657  1820  2966  1790  1970  2014  1994  1969  2130  2261  1943  2507
2015   1797  1944  2418  2133  2084  2250  2260  1987  2398  2328  2036  2579
2016   1790  2084  2366  2196  2284  2678  2114  2534  2352  2220  2320  2567
2017   2124  2139  2598  2232  2793  2851  2329  2639  2707  2583  2636  2799
2018   2358  2194  2636  2359  2526  2575  2502  2618  2449  2600  2616  2780
2019   2454  2215  2612  2592  2587  2494  2465  2489  2504  2668  2462  2830
2020   2453  2386  2652  2468  2354  2754  2581  2452  2804  2444  2295  2862
      Month  Count
Year              
2014     78  25021
2015     78  26214
2016     78  27505
2017     78  30430
2018     78  30213
2019     78  30372
2020     78  30505

True quantile of 30505: 0.220447
Commencing Mainsecition D
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month   1    2    3    4    5    6    7    8    9    10   11   12
Year                                                             
2014   127  123  157   99  127  106  123  120  115  157  115  170
2015   131  122  165  135  131  143  160  123  138  120  124  135
2016   111  106  143  153  108  173  145  122  164  162  181  161
2017   150  176  176  160  194  169  165  170  163  194  169  210
2018   181  155  207  200  168  188  160  180  185  148  204  182
2019   176  161  199  180  188  176  209  182  208  199  177  199
2020   202  162  194  177  152  155  182  167  181  161  195  188
      Month  Count
Year              
2014     78   1539
2015     78   1627
2016     78   1729
2017     78   2096
2018     78   2158
2019     78   2254
2020     78   2116

True quantile of 2116: 0.070665
Commencing Mainsecition E
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month   1    2    3    4    5    6    7    8    9    10   11   12
Year                                                             
2014   575  551  889  593  598  681  642  626  731  715  655  702
2015   558  637  722  688  649  711  699  701  661  717  679  780
2016   603  709  731  684  762  874  628  805  765  680  721  830
2017   684  744  894  695  836  886  647  804  806  806  788  916
2018   755  711  869  723  829  849  774  791  745  880  756  828
2019   773  768  807  857  845  807  814  807  826  888  780  838
2020   787  794  899  798  784  821  845  729  834  739  704  881
      Month  Count
Year              
2014     78   7958
2015     78   8202
2016     78   8792
2017     78   9506
2018     78   9510
2019     78   9810
2020     78   9615

True quantile of 9615: 0.071170
Commencing Mainsecition F
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month    1     2     3     4     5     6     7     8     9     10    11    12
Year                                                                         
2014   1566  1621  2280  1599  1653  1760  1797  1761  1945  2032  1709  2194
2015   1785  1693  2206  2007  1963  2116  1993  1877  2171  2153  2169  2337
2016   1746  1971  2238  1996  2021  2295  1977  2267  2307  2092  2223  2477
2017   2119  2081  2593  2010  2213  2477  2067  2178  2223  2179  2292  2495
2018   2219  1912  2489  2058  2100  2040  2119  2332  2149  2314  2257  2368
2019   2206  1941  2218  2226  2103  2108  2095  2008  2173  2226  2082  2194
2020   2059  1960  2173  1846  1678  1834  1770  1652  2058  1714  1742  2069
      Month  Count
Year              
2014     78  21917
2015     78  24470
2016     78  25610
2017     78  26927
2018     78  26357
2019     78  25580
2020     78  22555

True quantile of 22555: 0.045685
Commencing Mainsecition G
/home/sha/data/transfer/src/basic_analysis_numbers.py:124: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df_reduced["ms_present"] = df_reduced["mainsections"].str.contains(mainsection)
Month     1     2      3      4 

"""
