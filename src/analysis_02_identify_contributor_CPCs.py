import scipy
import scipy.sparse
import numpy as np
import pandas as pd
import pickle
import os
import pdb

from settings import DATA_DIR

""" Load Covid-19 mentioning shares (direct Covid-19 relevance)"""
#codes = pd.read_pickle("/home/sha/data/Covid19PatentApplications/data/work/df_Covid-19_subclasses_shares.pkl.gz", compression="gzip")
codes = pd.read_pickle(os.path.join(DATA_DIR, "work", "df_Covid-19_terms_subclasses_shares_fulltext_weighted_reduced.pkl.gz"), compression="gzip")

""" Load CPC codes -- file is misnamed, this is subsection"""
""" "/mnt/usba2/GreenPatentsProject/CPC/patent_classification_codes_level_group.pkl" is also in /home/sha/data/Covid19PatentApplications/tmp/patent_classification_codes_level_group.pkl"""
#with open("/mnt/usba2/GreenPatentsProject/CPC/patent_classification_codes_level_group.pkl", "rb") as rfile:
#with open("/home/sha/data/Covid19PatentApplications/tmp/patent_classification_codes_level_group.pkl", "rb") as rfile:
with open(os.path.join(DATA_DIR, "..", "tmp", "patent_classification_codes_level_group.pkl"), "rb") as rfile:
    codes_list = pickle.load(rfile)

pdb.set_trace()

"""Sort Covid-19 relatedness df in the same order as matrix"""
codes_order = pd.DataFrame({"Classification": codes_list, "sort_column": np.arange(len(codes_list))})
codes_ordered = pd.merge(codes, codes_order, on="Classification", how="inner")
print(len(codes_ordered))

codes_ordered.sort_values(by=["sort_column"], inplace=True)

""" Load matrix"""
#mtx = scipy.sparse.load_npz("/home/sha/tmp/cpc_citation_matrix_subclasses.npz")
mtx = scipy.sparse.load_npz(os.path.join(DATA_DIR, "..", "tmp", "cpc_citation_matrix_subclasses.npz"))

""" Remove codes not present in df from matrix, so they have the same number (and sequence) of codes"""
#if (len(codes_ordered) != 663):
#    print("Codes were lost. This script will fail.")
#codes_all = pd.merge(codes, codes_order, on="Classification", how="outer")
#print(codes_all)
#"""missing are C10H and F21H. Find indices to remove these from mtx"""
#drop_indices = np.where((codes_list=="C10H") | (codes_list=="F21H"))[0]
#"""missing is only F21H. Find indices to remove these from mtx"""
#drop_indices = np.where((codes_list=="F21H"))[0]
"""missing are F17B, C10H, and F21H. Find indices to remove these from mtx"""
drop_indices = np.where((codes_list=="F17B") | (codes_list=="C10H") | (codes_list=="F21H"))[0]
keep_indices = list(set(np.arange(len(codes_list)))-set(drop_indices)) 
mtx = mtx[keep_indices, :].tocsc()[:, keep_indices].tocsr()
codes_list_reduced = codes_list[keep_indices]

#mtx_dense = mtx.todense()
#>>> mtx_dense.shape
#(663, 663)


#mtx_weighted = mtx/mtx.sum(axis=0)
#np.sum(mtx_weighted > 1/mtx_weighted.shape[0])
#mtx_above_avg = mtx_weighted > 1/mtx_weighted.shape[0]
"""
# examples with full mtx, without dropped C10H and F21H
>>> np.sum(mtx_weighted > 1/mtx_weighted.shape[0], axis=1).T
matrix([[659, 662, 647,  11, 620, 637,   1, 653, 662, 658, 282,   2, 652,
         579, 656,   0, 657, 662, 662, 658, 528,   1,   3,  55, 640, 454,
           0, 661, 661,   8,  89,   0, 658, 660, 659,   4,   0, 660, 659,
          10,   0, 654, 661,   0, 556,   1, 652, 650, 662,  50, 557, 662,
         402, 662, 661,   3, 650,   2, 646,   0,   3, 661,   7, 654, 659,
         660, 657, 611, 646,  47, 659,   2, 656, 224, 624,   0,   2,  12,
         659,   1,   2,   0,   2, 588,   1,  29,   2,   1,   2,   4,   2,
           4, 248,   1, 643, 650, 658, 650,   0, 604, 661,   0,  12, 548,
           1, 653, 218,   0, 662, 229, 646,   1, 662,   0,   1,  11,   0,
           0, 662,   2,   1,   2, 658,   2,   0,   3,   2, 658, 637, 520,
         648,   8, 561, 450, 453,   1,   2,   4,  12,   2,  18,   2, 652,
           4,  40,   4,  12,  10,   7, 105,   1, 659, 654,   3,  12, 660,
           0,   3, 653,  90,   0,  14, 640, 659, 275,   0,   5,   1,   4,
           1,   3,   0,   0,   2, 537,   2, 157,   0, 660,  11,   2,   3,
           9,   0,   0,   0,   0, 658, 565,  19,  14,   1,   1,   0,  23,
           2,  29, 652,   2,   1, 113, 322,  15,   3,   4, 602,   0,   1,
           1,   0,   1, 656,   6, 500,   4,   0,  30,   1,   1,   0,   7,
           0,  10,   8,   0,   0,   3, 654,  23,  23, 576, 645,   3, 658,
         653,   4, 657,  72,   0,   5,  22, 558,   1,   0, 654, 659,  84,
          31,   6,   4,   3,   7, 583,   5,   0,   0, 647,  62,  19,  29,
           0,   1,   1, 651,  24, 659,   0, 653, 652,   6, 650, 654, 153,
           0,   2, 653,   2,   4,   0,   3,   2,   0,   4,   3,   0, 661,
           4, 646,   3,   9, 659, 660, 659, 660, 576, 647,   1,   1,   3,
           2, 663, 636,  47,  10,   1, 554,   1,   0,   0, 656,   1,   0,
         526,   0,  22, 175, 652,  21, 515,   0, 658,   1,   0,  13,   0,
           4,   0,   0,   5,   0,   0,   4,   0,   1,   0,   0,   0,   1,
           0,   0, 445,   0,   1,   7,   1, 656,  47,   4,   4,   4,   1,
         640,   1,   0,   3,   2,   1, 657, 662, 656,   0,   1,   0, 661,
           1, 653,   2,   3,   2, 654,   3,   3,   2,  19,   1,   1,  10,
           9,   3,   0,   1,   2,   1,   7,   1,  19, 534, 111,   2,   0,
           0,   1, 627,   2,  44,   1,   2,   0,  25, 651,   3,   5, 657,
           1,   2,   2,   3,   0,   1,   2,   0,   2,   0,   1,   1,   0,
           0,   3,   4, 656,   2,   3,   0,   0,   8,   9,   0,   0,   5,
           0, 658, 619, 641,   0,   3, 628,  11,  20,   7,  11, 121, 659,
          50, 648,   0,  19,   2,   4,   1, 653,   0,   0,   1,   0,   2,
           6,   5, 663,   2,   0,   0,   0,   0,   6,   1,   1, 660,   1,
           2,   7,   7,   0,   0,   4,   0,  12, 655,   1,   0,   0,   0,
          30,   0,  38,   2,   5,   0,   0,   0, 378,   0,   2,   0,   0,
           1,   4,   0,   0,   2,   2,   2,   0,   0,   0,   0,   0,  12,
         584,   0,   2,   1,   5,   0, 143,   4,   1,   0,   0,   1,   0,
           6, 642, 659, 653,   2,   1,   0, 152,   2,   1,   2,   5, 658,
           6,   0,   0,   2,   0,   0,   0,   1,   1, 415,   1,   0,   0,
           0,   0,   4,   0,   0,   0,   1,   0,   3,   0,   0,   2,   2,
          55,   2,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
           0,   2,   0,   1,   1,   0,   0,   0,   1,   0,   1,   2,  27,
           3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          14,   0,   0,   0,   0,   0,   0,   1,   0,   1,   0,   0,   0,
           0,   7,   0,   0,   0,   0,   1,   0,   1,   0,  23,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,
           0,   0,   0,   4,   2,   0,   0,   1,   0,   0,   0,   0,   0]])
>>> np.sum(mtx_weighted > 1/mtx_weighted.shape[0], axis=0)
matrix([[153, 151, 151, 155, 151, 150, 148, 150, 151, 150, 150, 147, 152,
         153, 149, 148, 153, 151, 151, 152, 152, 151, 150, 151, 148, 149,
         153, 151, 151, 148, 148, 149, 150, 150, 151, 154, 153, 151, 150,
         151, 149, 149, 151, 149, 148, 151, 150, 150, 151, 152, 150, 151,
         151, 151, 151, 150, 150, 157, 150, 148, 151, 148, 151, 150, 152,
         150, 151, 151, 151, 150, 150, 150, 148, 150, 152, 148, 150, 150,
         150, 146, 153, 151, 153, 152, 150, 152, 149, 153, 153, 149, 147,
         153, 151, 147, 148, 153, 152, 152, 150, 148, 152, 144, 154, 152,
         151, 151, 152, 150, 151, 153, 154, 149, 152, 155, 147, 150, 155,
         144, 153, 151, 148, 154, 152, 154, 155, 149, 149, 151, 152, 152,
         153, 148, 150, 153, 148, 150, 152, 148, 145, 151, 148, 146, 151,
         152, 152, 149, 152, 150, 153, 151, 150, 150, 153, 150, 151, 152,
         153, 149, 153, 154, 153, 151, 154, 153, 150, 146, 148, 147, 151,
         147, 149, 146, 148, 148, 148, 149, 152, 147, 152, 152, 151, 153,
         150, 149, 145, 149, 152, 150, 150, 151, 149, 148, 152, 147, 151,
         146, 150, 153, 151, 152, 153, 151, 149, 148, 148, 150, 152, 152,
         151, 149, 156, 153, 152, 149, 145, 151, 150, 151, 155, 140, 151,
         155, 148, 151, 150, 144, 144, 150, 151, 150, 152, 151, 154, 152,
         151, 151, 151, 151, 154, 154, 151, 151, 146, 158, 149, 150, 148,
         151, 150, 157, 154, 152, 150, 148, 158, 155, 151, 153, 151, 150,
         159, 147, 151, 151, 149, 152, 146, 149, 151, 156, 154, 149, 149,
         155, 147, 151, 149, 156, 144, 156, 144, 153, 150, 150, 152, 151,
         152, 149, 149, 151, 151, 153, 152, 150, 147, 154, 149, 149, 147,
         145, 152, 151, 152, 151, 151, 153, 149, 151, 146, 152, 153, 150,
         149, 149, 151, 154, 151, 153, 153, 151, 150, 148, 149, 152, 167,
         151, 148, 156, 152, 147, 149, 149, 148, 151, 144, 149, 158, 148,
         151, 154, 146, 147, 153, 153, 147, 150, 151, 150, 150, 147, 148,
         151, 152, 151, 149, 150, 146, 152, 151, 152, 154, 155, 151, 151,
         145, 151, 152, 153, 143, 154, 149, 152, 150, 148, 151, 157, 149,
         148, 150, 153, 150, 143, 145, 150, 152, 150, 152, 151, 150, 150,
         148, 149, 150, 153, 155, 151, 149, 151, 151, 152, 146, 145, 149,
         151, 156, 149, 152, 149, 150, 146, 155, 144, 155, 145, 153, 147,
         150, 152, 148, 150, 152, 148, 154, 152, 151, 151, 154, 154, 151,
         148, 151, 150, 148, 153, 145, 152, 152, 151, 150, 150, 148, 151,
         150, 149, 152, 149, 147, 151, 154, 150, 154, 150, 145, 148, 157,
         150, 147, 151, 151, 155, 145, 151, 153, 152, 149, 147, 151, 156,
         148, 153, 149, 156, 148, 151, 170, 149, 153, 149, 148, 154, 147,
         147, 152, 151, 152, 152, 150, 157, 151, 153, 152, 151, 149, 151,
         144, 149, 152, 144, 148, 150, 151, 149, 152, 202, 155, 150, 153,
         152, 154, 147, 149, 151, 149, 154, 150, 149, 153, 147, 150, 146,
         149, 151, 150, 151, 145, 153, 159, 145, 152, 145, 151, 151, 152,
         152, 150, 147, 149, 147, 146, 152, 148, 147, 152, 147, 149, 142,
         156, 150, 153, 150, 147, 151, 150, 150, 153, 149, 152, 150, 150,
         150, 146, 156, 150, 149, 149, 144, 147, 148, 148, 148, 152, 153,
         163, 153, 141, 143, 146, 147, 164, 156, 156, 148, 147, 151, 149,
         151, 141, 151, 148, 153, 152, 156, 198, 158, 145, 149, 149, 149,
         152, 155, 151, 164, 146, 148, 153, 155, 149, 156, 151, 152, 118,
         157, 148, 153, 157, 118, 151, 151, 144, 150, 148, 150, 151, 152,
         149, 150, 183, 151, 156, 162, 152, 190, 176, 152, 154,  97, 154,
         147, 190, 152, 177, 146, 149, 175, 148, 147, 160, 158, 184, 152,
         142,  92, 105, 150, 151, 144, 119, 145, 149, 117, 156,  47,  11]])
"""

""" Compute row shares"""
mtx_weighted = mtx/mtx.sum(axis=0)
""" row average has to be 1/n for each row"""
row_average = 1/mtx_weighted.shape[0]

""" get moments"""
""" get all matrix entries as flat 1-d array"""
mtx_weighted_sample = np.asarray(mtx_weighted.flatten())[0]
""" compute mean + std. This is a scalar value."""
mean_plus_std = np.mean(mtx_weighted_sample) + np.std(mtx_weighted_sample)

#pdb.set_trace()
""" for two different thresholds ... """
for above_avg_threshold in [row_average, mean_plus_std]:
    
    """ obtain bool matrix of which entries are above average"""
    #mtx_above_avg = mtx_weighted > 1/mtx_weighted.shape[0]
    mtx_above_avg = mtx_weighted > above_avg_threshold

    """ check size, set desired number?"""
    mtx_size = mtx_above_avg.shape[0] # 661
    highes_val_number = 50

    """ find highest 50 values of "Share" """
    #threshold = np.sort(codes_ordered["Share"])[-50]
    threshold = np.sort(codes_ordered["Share"])[-highes_val_number]

    """ set those 1 on target column, all others 0 """
    codes_ordered["covid_r_vector"] = 0
    codes_ordered.loc[codes_ordered["Share"]>=threshold,"covid_r_vector"] = 1

    #pdb.set_trace()
    """ multiply target column by mtx_above_avg.T"""
    vector_c = np.dot(codes_ordered["covid_r_vector"], mtx_above_avg.T)
    vector_c1 = vector_c > 0
    #np.sum(vector_c1, axis=1)
    #matrix([[190]])

    """ relativly more citations from these subclasses to covid-related than to others"""
    vector_all = np.dot(np.ones(mtx_size), mtx_above_avg.T) * highes_val_number / mtx_size
    vector_c2 = vector_c > vector_all*1.05
    #np.sum(vector_c2, axis=1)
    #matrix([[35]])

    """ find values of target that are 1"""
    indices_c1 = np.where(vector_c1)[1]
    codes_c1 = codes_list_reduced[indices_c1]

    indices_c2 = np.where(vector_c2)[1]
    codes_c2 = codes_list_reduced[indices_c2]

    #pdb.set_trace()
    """ make these into data frame of len 50""" 
    if above_avg_threshold == row_average:
        codes_c2_df = pd.DataFrame({"Classification": codes_c2, "Share": np.zeros(len(codes_c2))})
        codes_c2_df.to_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_relative_35.pkl.gz"), compression="gzip")
    elif above_avg_threshold == mean_plus_std:
        #print(np.sum(vector_c1, axis=1))
        print(codes_c1)
        codes_c1_df = pd.DataFrame({"Classification": codes_c1, "Share": np.zeros(len(codes_c1))})
        codes_c1_df.to_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_abs_43.pkl.gz"), compression="gzip")

""" compute direct flow of shares (backwards) through citation matrix"""
codes_c3_df = codes_ordered[['Classification', "Share"]]
codes_c3_df.columns = ['Classification', 'Direct_Covid_relevance']
codes_c3_df['Indirect_Covid_relevance'] = np.array(np.dot(np.matrix(codes_ordered["Share"]), mtx_weighted.T))[0]
codes_c3_df.to_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_shares.pkl.gz"), compression="gzip")
"""
mps = np.mean(codes_c3_df['Indirect_Covid_relevance']) + np.std(codes_c3_df['Indirect_Covid_relevance'])
print(codes_c3_df.loc[codes_c3_df['Indirect_Covid_relevance']>mps,:])
    Classification  Direct_Covid_relevance  Indirect_Covid_relevance
160           H01L                0.000010                  0.003227
38            A61B                0.000504                  0.001426
284           Y10S                0.000000                  0.003399
117           Y10T                0.000059                  0.005061
133           G02B                0.000036                  0.001208
28            A61M                0.000803                  0.000633
150           G01R                0.000020                  0.000653
14            G01N                0.002250                  0.001375
158           B29C                0.000013                  0.000756
162           H04B                0.000008                  0.000793
77            G06K                0.000165                  0.000695
137           H04L                0.000033                  0.002402
119           G06F                0.000057                  0.003931
164           G11C                0.000007                  0.000769
113           H04W                0.000061                  0.001061
103           B65D                0.000077                  0.000623
128           G11B                0.000041                  0.001073
292           H05K                0.000000                  0.000774
161           H04N                0.000010                  0.001860
165           C08L                0.000005                  0.000792
85            A61F                0.000132                  0.000683
362           G03G                0.000000                  0.000586
251           H01R                0.000000                  0.000657
63            Y02P                0.000214                  0.000617
288           Y02E                0.000000                  0.000623
96            G06T                0.000098                  0.000627
78            G06Q                0.000165                  0.000970
641           C08G                0.000000                  0.000593
68            B01J                0.000198                  0.000631
287           Y02T                0.000000                  0.000677
50            B01D                0.000325                  0.000704
19            C07D                0.001417                  0.001180
53            C07C                0.000281                  0.000893
8             C12N                0.003385                  0.000622
2             A61K                0.004703                  0.001720
10            C07K                0.003295                  0.000697
139           H04M                0.000031                  0.000675
"""

pdb.set_trace()
threshold = np.sort(codes_c3_df["Indirect_Covid_relevance"])[-50]
codes_c4_df = codes_c3_df[codes_c3_df["Indirect_Covid_relevance"]>=threshold]
codes_c4_df = codes_c4_df[['Classification', 'Indirect_Covid_relevance']]
codes_c4_df = codes_c4_df.sort_values(by="Indirect_Covid_relevance",ascending=False)
codes_c4_df.columns = ['Classification', 'Share']
codes_c4_df.to_pickle(os.path.join(DATA_DIR, "work", "df_upstream_from_Covid-19_abs_50.pkl.gz"), compression="gzip")
""" Print highest 50 of indirect relevance in percentage"""
codes_c4_df["Share"] = codes_c4_df["Share"]*100
codes_c4_df.columns = ['Classification', 'Share in Percent']
print(codes_c4_df)


""" Print highest 50 of direct relevance"""
threshold_direct = np.sort(codes["Share"])[-50]
codes = codes.loc[codes["Share"]>=threshold_direct]
codes["Share"] = codes["Share"]*100
codes.columns = ['Classification', 'Share in Percent']
print(codes)

print(sum(codes_c4_df["Share in Percent"]))
print(sum(codes["Share in Percent"]))
pdb.set_trace()
