import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
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
df_cov_upstream = pd.read_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_abs_50.pkl.gz"), compression='gzip') 
df = pd.merge(df, df_cov, on="Classification", how='left')
df["Covid_mentioned"] = 0
df.loc[df["Share"]>0.00041, "Covid_mentioned"] = 1
del df["Share"]
del df_cov_upstream["Share"]
df_cov_upstream["Upstream_from_Covid_mentioned"] = 1
df = pd.merge(df, df_cov_upstream, on="Classification", how='left')
df.loc[df["Upstream_from_Covid_mentioned"].isnull(), "Upstream_from_Covid_mentioned"] = 0
df["Upstream_from_Covid_mentioned"] = df["Upstream_from_Covid_mentioned"].apply(int)

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
print(df)

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0][0].fill_between(df.index, df["MinGrowth"], df["MaxGrowth"], color='b', alpha=0.3)
for cn in growth_columns_covid:
    ax[0][0].scatter(df.index, df[cn], c=df["col"], s=5)
ax[0][0].plot(df.index, df.index*0, c='k', linewidth=0.5, linestyle="--")
ax[0][0].plot(df.index, df["MedGrowth"], c='k', linewidth=0.5)
ax[0][0].set_ylabel("Annual growth")
ax[0][0].set_ylim(-1, 3)
ax[0][0].set_xlim(0, 667)
ax[0][0].set_xticks([40, 160, 300, 366, 402, 480, 560, 632, 662], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y'])
plt.savefig(os.path.join(OUTPUT_DIR, "subclass_growth_comparison_precovid_average.pdf"))


mainsections = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']
for ms in mainsections:
    dfs = df[df["Mainsection"]==ms]
    dfs = dfs[~dfs["Growth_2020"].isnull()]
    print("Mainsection ", ms, " Top: ", list(dfs['Classification'][:5]), " Bottom: ", list(dfs['Classification'][-5:]))
    print("     Growth: ", list(dfs['Growth_2020'][:5]), " - ", list(dfs['Growth_2020'][-5:]))

"""
# [668 rows x 29 columns]
# Mainsection  A  Top:  ['A62B', 'A22C', 'A23Y', 'A24C', 'A23D']  Bottom:  ['A63G', 'A44B', 'A47F', 'A23F', 'A21B']
     # Growth:  [0.34234234234234234, 0.2647058823529412, 0.4, 0.325, 0.37735849056603776]  -  [-0.31007751937984496, -0.1693121693121693, -0.1939799331103679, -0.1188118811881188, -0.3333333333333333]
# Mainsection  B  Top:  ['B21L', 'B31C', 'B62H', 'B27B', 'B28C']  Bottom:  ['B60T', 'B09C', 'B61H', 'B43K', 'B68F']
     # Growth:  [0.0, 0.5, 0.5238095238095238, 0.3176470588235294, 0.375]  -  [-0.28527131782945736, -0.2833333333333333, -0.375, -0.5362318840579711, 0.0]
# Mainsection  C  Top:  ['C12J', 'C07G', 'C06D', 'C12H', 'C13K']  Bottom:  ['C25B', 'C23C', 'C09J', 'C05B', 'C04B']
     # Growth:  [8.0, 0.5714285714285714, 0.6, 0.2903225806451613, 0.03125]  -  [-0.2084942084942085, -0.16358695652173913, -0.06027820710973725, -0.5454545454545454, -0.1822600243013366]
# Mainsection  D  Top:  ['D06L', 'D21F', 'D03J', 'D01G', 'D02J']  Bottom:  ['D06B', 'D03D', 'D01C', 'D01F', 'D06G']
     # Growth:  [2.6, 0.3088235294117647, 1.0, 0.21428571428571427, 0.6153846153846154]  -  [-0.2894736842105263, -0.18181818181818182, -0.4, -0.236, -0.6666666666666666]
# Mainsection  E  Top:  ['E02C', 'E02F', 'E04H', 'E03B', 'E21D']  Bottom:  ['E21B', 'E05D', 'E01F', 'E06C', 'E05B']
     # Growth:  [1.0, 0.002617801047120419, -0.03316749585406302, 0.2268041237113402, -0.1702127659574468]  -  [-0.1288681644764731, -0.28391167192429023, -0.21705426356589147, -0.27358490566037735, -0.140625]
# Mainsection  F  Top:  ['F03D', 'F23J', 'F28C', 'F25J', 'F23K']  Bottom:  ['F01K', 'F15B', 'F28D', 'F24B', 'F04C']
     # Growth:  [0.14093959731543623, 0.15151515151515152, 0.11538461538461539, 0.2111111111111111, 0.21739130434782608]  -  [-0.3783783783783784, -0.2661290322580645, -0.19433962264150945, -0.2830188679245283, -0.18811881188118812]
# Mainsection  G  Top:  ['G01Q', 'G10D', 'G06M', 'G03H', 'G21C']  Bottom:  ['G06C', 'G16C', 'G16Y', 'G06J', 'G16Z']
     # Growth:  [0.3181818181818182, 0.2268041237113402, 0.5384615384615384, 0.23214285714285715, 0.043795620437956206]  -  [-0.75, -0.1875, 1.5568181818181819, 0.5, 0.6571428571428571]
# Mainsection  H  Top:  ['H05F', 'H05C', 'H01C', 'H02S', 'H03J']  Bottom:  ['H01Q', 'H01M', 'H05K', 'H02M', 'H03M']
     # Growth:  [-0.10714285714285714, 1.0, -0.06306306306306306, 0.03469387755102041, 0.21052631578947367]  -  [-0.10255281690140845, -0.14794338788146838, -0.15720425243157657, -0.154, -0.23537702607470048]
# Mainsection  Y  Top:  ['Y10S', 'Y02C', 'Y02B', 'Y10T', 'Y04S']  Bottom:  ['Y02E', 'Y02W', 'Y02T', 'Y02P', 'Y02D']
     # Growth:  [-0.3304157549234136, 0.13513513513513514, -0.345679012345679, -0.23096446700507614, -0.18681318681318682]  -  [-0.26374859708193044, -0.2, -0.4864864864864865, -0.3989769820971867, -0.3856707317073171]

    Classification  Total  2014_Count  2014_Length  2015_Count  2015_Length  2016_Count  2016_Length  2017_Count  ...  MinGrowth  AvgGrowth  MedGrowth          Sum  Sum_pre_covid  logSum_pre_covid  Mainsection      col  relative_g_2020
0             A41D      0  577.965873        365.0  620.065812        366.0  615.955231        365.0  563.299939  ...  -0.085486  -0.007646  -0.006629  4131.051970    3454.402151          8.147405            A  #e6194b        -1.981860
1             A62B      0  182.926190        365.0  185.409524        366.0  195.249206        365.0  169.863095  ...  -0.130019   0.001506   0.033579  1311.983730    1091.771825          6.995557            A  #e6194b        -1.852518
2             A45B      0   57.366667        365.0   66.258333        366.0   59.729762        365.0   69.059524  ...  -0.259775   0.002210   0.058162   446.878571     379.583333          5.939074            A  #e6194b        -1.211237
3             A47H      0   40.950000        365.0   34.500000        366.0   31.850000        365.0   41.259524  ...  -0.157509  -0.012905  -0.076812   273.203968     226.203968          5.421437            A  #e6194b        -1.020127
4             A41F      0   56.953968        365.0   53.545635        366.0   82.120238        365.0   71.288889  ...  -0.254131  -0.010201  -0.131896   422.603247     371.096898          5.916463            A  #e6194b        -0.480971
..             ...    ...         ...          ...         ...          ...         ...          ...         ...  ...        ...        ...        ...          ...            ...               ...          ...      ...              ...
665           Y02A      0         NaN          NaN    0.125000        366.0    1.642857        365.0   56.245635  ...  -0.069969  12.089943   7.596634   672.058308     497.709102          6.210016            Y  #000075         0.003217
666           Y02E      0    7.850000        365.0   10.400000        366.0   16.737302        365.0  314.665109  ...  -0.156322   3.710284   0.324841  1111.831269     914.313339          6.818173            Y  #000075         0.004414
667           Y02D      0         NaN          NaN    0.500000        366.0    2.000000        365.0   68.189683  ...  -0.186266   9.780008   3.105728   737.168382     591.554060          6.382753            Y  #000075         0.005727
668           Y02T      0    4.200000        365.0    2.666667        366.0   10.472222        365.0  179.562229  ...  -0.365079   3.733000   0.127403   651.912062     567.175734          6.340669            Y  #000075         0.007876
669           Y02P      0    0.533333        365.0    1.866667        366.0   10.693687        365.0  190.043312  ...  -0.009696   4.808569   2.500000   732.685362     601.140088          6.398828            Y  #000075         0.019429

[670 rows x 34 columns]
findfont: Font family ['normal'] not found. Falling back to DejaVu Sans.
Mainsection  A  Top:  ['A41D', 'A62B', 'A45B', 'A47H', 'A41F']  Bottom:  ['A45D', 'A61J', 'A61Q', 'A61M', 'A23L']
     Growth:  [0.22829669946471376, 0.20915669266118017, 0.2440688410581451, 0.30454895913646873, 0.1247681935562141]  -  [-0.3298657649229108, -0.28190947782187287, -0.4022683520906466, -0.18448636627206502, -0.37958259241510844]
Mainsection  B  Top:  ['B68F', 'B64B', 'B60V', 'B82B', 'B25F']  Bottom:  ['B65B', 'B01D', 'B01J', 'B24B', 'B62C']
     Growth:  [1.6666666666666665, 0.8426981657806117, 0.6274509803921569, 0.498019801980198, 0.07420245221408335]  -  [-0.2973797089744544, -0.3294342702231827, -0.387710612337105, -0.33967593374262217, -0.33333333333333337]
Mainsection  C  Top:  ['C12J', 'C07G', 'C06B', 'C09F', 'C06D']  Bottom:  ['C08G', 'C09J', 'C12Q', 'C07K', 'C12N']
     Growth:  [1.9333333333333331, -0.10163174775933574, -0.17816527672479154, -0.12162162162162167, -0.01304347826086952]  -  [-0.46284335206966326, -0.4834787371503947, -0.23749131188534087, -0.2795985884561937, -0.31851571017268593]
Mainsection  D  Top:  ['D06L', 'D03J', 'D21G', 'D05B', 'D04D']  Bottom:  ['D10B', 'D06F', 'D06H', 'D21F', 'D01F']
     Growth:  [0.8186988171064603, 1.6698935140367859, -0.06960330980773904, -0.21232636862378892, -0.33333333333333337]  -  [-0.40917976726905964, -0.3183628846894837, -0.5144490136601454, -0.3322374250637074, -0.5322964016026168]
Mainsection  E  Top:  ['E02C', 'E01D', 'E03B', 'E05G', 'E03C']  Bottom:  ['E04B', 'E05D', 'E02D', 'E04C', 'E01C']
     Growth:  [2.0, 0.1858407079646018, 0.04310039237736689, -0.28016034614405694, -0.14212676064251162]  -  [-0.26178058039172003, -0.3367273107018447, -0.301589471367266, -0.39738239018665167, -0.3515661835822912]
Mainsection  F  Top:  ['F41C', 'F23H', 'F25J', 'F23J', 'F02G']  Bottom:  ['F16C', 'F16K', 'F15B', 'F16D', 'F28D']
     Growth:  [0.08409294104348011, 1.7999999999999998, 0.022453108219486023, -0.11819861613461227, -0.011740041928721221]  -  [-0.3877583035531123, -0.32997236949281705, -0.4754469862517198, -0.4266138974452724, -0.3453767055927034]
Mainsection  G  Top:  ['G10D', 'G06M', 'G21C', 'G21H', 'G11C']  Bottom:  ['G03F', 'G01N', 'G01F', 'G07C', 'G16Z']
     Growth:  [0.12349274473737996, 0.4312114989733059, 0.05204856919690408, -0.1333333333333334, -0.015773379140052435]  -  [-0.2537602395514338, -0.29738348347487953, -0.35101426958887666, -0.2436758514969336, 0.6401677220605855]
Mainsection  H  Top:  ['H03J', 'H05C', 'H05F', 'H04K', 'H04H']  Bottom:  ['H01M', 'H01S', 'H02P', 'H02K', 'H01R']
     Growth:  [0.4280195201301343, 0.0, -0.21115409413281755, -0.08482340248669107, -0.19765441782449658]  -  [-0.3313447637110367, -0.2724673290081294, -0.3664723728007975, -0.3637677733570301, -0.19318844625340784]
Mainsection  Y  Top:  ['Y10S', 'Y02W', 'Y02C', 'Y02B', 'Y04S']  Bottom:  ['Y02A', 'Y02E', 'Y02D', 'Y02T', 'Y02P']
     Growth:  [-0.3245841640584706, -0.20077122554308902, 0.06876286730301334, -0.3721794290040958, -0.14247853745297448]  -  [-0.17712344156860116, -0.2355886774582304, -0.3768817760378153, -0.49512300365241396, -0.33573761893399784]
    Classification  Total  2014_Count  2014_Length  2015_Count  2015_Length  2016_Count  2016_Length  2017_Count  ...  AvgGrowth  MedGrowth          Sum  Sum_pre_covid  logSum_pre_covid  Mainsection      col  relative_g_2020  invGrowth_2020
0             A47H      0   40.950000        365.0   34.500000        366.0   31.850000        365.0   41.259524  ...  -0.012905  -0.076812   273.203968     226.203968          5.421437            A  #e6194b        -1.020127       -0.304549
1             A45B      0   57.366667        365.0   66.258333        366.0   59.729762        365.0   69.059524  ...   0.002210   0.058162   446.878571     379.583333          5.939074            A  #e6194b        -1.211237       -0.244069
2             A41D      0  577.965873        365.0  620.065812        366.0  615.955231        365.0  563.299939  ...  -0.007646  -0.006629  4131.051970    3454.402151          8.147405            A  #e6194b        -1.981860       -0.228297
3             A62B      0  182.926190        365.0  185.409524        366.0  195.249206        365.0  169.863095  ...   0.001506   0.033579  1311.983730    1091.771825          6.995557            A  #e6194b        -1.852518       -0.209157
4             A41F      0   56.953968        365.0   53.545635        366.0   82.120238        365.0   71.288889  ...  -0.010201  -0.131896   422.603247     371.096898          5.916463            A  #e6194b        -0.480971       -0.124768
..             ...    ...         ...          ...         ...          ...         ...          ...         ...  ...        ...        ...          ...            ...               ...          ...      ...              ...             ...
665           Y10S      0  578.180556        365.0  520.782143        366.0  355.582179        365.0  465.665331  ...  -0.205106  -0.317215  2389.385097    2297.592600          7.739617            Y  #000075        -0.198290        0.324584
666           Y02P      0    0.533333        365.0    1.866667        366.0   10.693687        365.0  190.043312  ...   4.808569   2.500000   732.685362     601.140088          6.398828            Y  #000075         0.019429        0.335738
667           Y02B      0    4.083333        365.0    7.392857        366.0   13.601190        365.0  290.211580  ...   4.222211   0.810496   592.972702     537.358777          6.286666            Y  #000075        -0.008040        0.372179
668           Y02D      0         NaN          NaN    0.500000        366.0    2.000000        365.0   68.189683  ...   9.780008   3.105728   737.168382     591.554060          6.382753            Y  #000075         0.005727        0.376882
669           Y02T      0    4.200000        365.0    2.666667        366.0   10.472222        365.0  179.562229  ...   3.733000   0.127403   651.912062     567.175734          6.340669            Y  #000075         0.007876        0.495123

[670 rows x 35 columns]
Mainsection  A  Top:  ['A47H', 'A45B', 'A41D', 'A62B', 'A41F', 'A43D', 'A61L', 'A01H', 'A41C', 'A61H']  Bottom:  ['A01J', 'A23F', 'A21B', 'A01L', 'A01P']
     Growth:  [0.30454895913646873, 0.2440688410581451, 0.22829669946471376, 0.20915669266118017, 0.1247681935562141]  -  [-0.5016982047549733, -0.5034288636244164, -0.505002382086708, -0.6058394160583941, -0.6781566989489145]
Mainsection  B  Top:  ['B68F', 'B64B', 'B60V', 'B82B', 'B62H', 'B42P', 'B43M', 'B68B', 'B25F', 'B62L']  Bottom:  ['B42C', 'B27N', 'B21B', 'B21H', 'B27H']
     Growth:  [1.6666666666666665, 0.8426981657806117, 0.6274509803921569, 0.498019801980198, 0.2010779823659531]  -  [-0.5996788537549407, -0.6037831184815702, -0.605648218743146, -0.6163477912451167, -0.6666666666666666]
Mainsection  C  Top:  ['C12J', 'C06D', 'C07G', 'C09F', 'C11C', 'C30B', 'C40B', 'C09G', 'C06B', 'C12F']  Bottom:  ['C23G', 'C21D', 'C05B', 'C23D', 'C06C']
     Growth:  [1.9333333333333331, -0.01304347826086952, -0.10163174775933574, -0.12162162162162167, -0.1234116767710323]  -  [-0.643685923015532, -0.6439148038745175, -0.7386182571371863, -0.7984886649874056, -0.8]
Mainsection  D  Top:  ['D03J', 'D06L', 'D21G', 'D21B', 'D21D', 'D01H', 'D05C', 'D05B', 'D02G', 'D21C']  Bottom:  ['D01C', 'D07B', 'D02H', 'D03C', 'D06G']
     Growth:  [1.6698935140367859, 0.8186988171064603, -0.06960330980773904, -0.14616602907074366, -0.1766846361185984]  -  [-0.542483660130719, -0.546072186836518, -0.5833333333333334, -0.7727272727272727, -0.85]
Mainsection  E  Top:  ['E02C', 'E01D', 'E03B', 'E04H', 'E03C', 'E03F', 'E21B', 'E04D', 'E04G', 'E06B']  Bottom:  ['E04C', 'E21D', 'E01B', 'E21F', 'E21C']
     Growth:  [2.0, 0.1858407079646018, 0.04310039237736689, -0.09924196324847791, -0.14212676064251162]  -  [-0.39738239018665167, -0.40957781978575925, -0.5194557669082126, -0.5323834196891192, -0.5863605209047292]
Mainsection  F  Top:  ['F23H', 'F24V', 'F41C', 'F23K', 'F28G', 'F22G', 'F25J', 'F02G', 'F41F', 'F41G']  Bottom:  ['F22D', 'F25C', 'F16T', 'F27M', 'F03C']
     Growth:  [1.7999999999999998, 0.19507497592516157, 0.08409294104348011, 0.0684726109556177, 0.04378378378378372]  -  [-0.5120772946859904, -0.5784535926704303, -0.6, -0.6923076923076924, -0.8335195530726257]
Mainsection  G  Top:  ['G16Z', 'G16Y', 'G06M', 'G04D', 'G10D', 'G06J', 'G21C', 'G11C', 'G16H', 'G06N']  Bottom:  ['G10F', 'G03C', 'G12B', 'G21D', 'G21B']
     Growth:  [0.6401677220605855, 0.5471806674338321, 0.4312114989733059, 0.134453781512605, 0.12349274473737996]  -  [-0.5327868852459017, -0.5864077669902913, -0.6, -0.6072709878019613, -0.6224242424242424]
Mainsection  H  Top:  ['H03J', 'H05C', 'H03H', 'H04K', 'H03F', 'H03K', 'H02N', 'H03D', 'H04N', 'H01L']  Bottom:  ['H02K', 'H01H', 'H02P', 'H01T', 'H01K']
     Growth:  [0.4280195201301343, 0.0, -0.008994671317472972, -0.08482340248669107, -0.10352570602434862]  -  [-0.3637677733570301, -0.3649364055662603, -0.3664723728007975, -0.4809322033898305, -0.5309833024118739]
Mainsection  Y  Top:  ['Y02C', 'Y04S', 'Y02A', 'Y02W', 'Y02E', 'Y10T', 'Y10S', 'Y02P', 'Y02B', 'Y02D']  Bottom:  ['Y10S', 'Y02P', 'Y02B', 'Y02D', 'Y02T']
     Growth:  [0.06876286730301334, -0.14247853745297448, -0.17712344156860116, -0.20077122554308902, -0.2355886774582304]  -  [-0.3245841640584706, -0.33573761893399784, -0.3721794290040958, -0.3768817760378153, -0.49512300365241396]
> /home/sha/data/CovidPA2/src/subclass_growth_comparison_precovid_average.py(156)<module>()
-> models = [  "Growth_2020 ~ Covid_mentioned", # m0
"""

df["invGrowth_2020"] = -1 * df["Growth_2020"]
df.sort_values(by=["Mainsection","invGrowth_2020"], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0][0].fill_between(df.index, df["MinGrowth"], df["MaxGrowth"], color='b', alpha=0.3)
for cn in growth_columns_covid:
    ax[0][0].scatter(df.index, df[cn], c=df["col"], s=5)
ax[0][0].plot(df.index, df.index*0, c='k', linewidth=0.5, linestyle="--")
ax[0][0].plot(df.index, df["MedGrowth"], c='k', linewidth=0.5)
ax[0][0].set_ylabel("Annual growth")
ax[0][0].set_ylim(-1, 3)
ax[0][0].set_xlim(0, 667)
ax[0][0].set_xticks([40, 160, 300, 366, 402, 480, 560, 632, 662], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y'])
plt.savefig(os.path.join(OUTPUT_DIR, "subclass_growth_comparison_precovid_average_g2020only.pdf"))


mainsections = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']
for ms in mainsections:
    dfs = df[df["Mainsection"]==ms]
    dfs = dfs[~dfs["Growth_2020"].isnull()]
    print("Mainsection ", ms, " Top: ", list(dfs['Classification'][:10]), " Bottom: ", list(dfs['Classification'][-5:]))
    print("     Growth: ", list(dfs['Growth_2020'][:5]), " - ", list(dfs['Growth_2020'][-5:]))

