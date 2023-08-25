import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import gzip
import pickle
import glob
import os
import pdb

import analysis_helpers

from settings import DATA_DIR, OUTPUT_DIR

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

# TODO: move to helper file, is used identically in other scripts
def old_get_processed_files(directory=os.path.join(DATA_DIR, 'processed')):
    """Function for collecting filenames of already downloaded raw files.
    @param directory: place where the files are stored
    @return: list of zip files
    """
    return glob.glob(os.path.join(directory, '*.zip_patents.pkl.gz'))

# TODO: remove, not used here
def create_plot_old(df_plot, plot_column, output_directory, maximum_age):
    """ Function for creating plot of the number of patent applications in a certain category 
        (e.g. all or certain main groups).
        @param df_plot: pandas dataframe with the data in wide format (Day of the year vs Year)
        @param plot_column: Category that is currently being plotted
        @param output_directory: Where plots are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
    """
    df_plot["Day"] = df_plot.index
    df_plot.loc[df_plot.index!=366] = df_plot.loc[df_plot.index!=366].fillna(0)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
    ax[0][0].plot(df_plot["Day"], df_plot[2015], linewidth=0.8, color="#7777ff", label="2015")
    ax[0][0].plot(df_plot["Day"], df_plot[2016], linewidth=0.8, color="#3333ff", label="2016")
    ax[0][0].plot(df_plot["Day"], df_plot[2017], linewidth=0.8, color="#0000ff", label="2017")
    ax[0][0].plot(df_plot["Day"], df_plot[2018], linewidth=0.8, color="#0000aa", label="2018")
    ax[0][0].plot(df_plot["Day"], df_plot[2019], linewidth=0.8, color="#000077", label="2019")
    ax[0][0].plot(df_plot["Day"], df_plot[2020], linewidth=1.5, color="#ff0000", label="2020")
    ax[0][0].plot(df_plot["Day"], df_plot[2021], linewidth=1.5, color="#aa0000", label="2021")
    ax[0][0].legend(loc='best')
    ax[0][0].set_ylabel("# Patent applications")
    ax[0][0].set_xlim(1, 366)
    ax[0][0].xaxis.set_ticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    ax[0][0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "patent_applications_" + plot_column + '_maximum_age_' + str(maximum_age) + ".pdf"))
    #plt.show()

# TODO: remove (not used here) or move to helper file, is used identically in other scripts
def old_find_code_in_PA(pa_codes, code_of_interest):
    code_present = False
    pa_codes_idx = 0
    while (not code_present) and pa_codes_idx < len(pa_codes):
        if code_of_interest == pa_codes[pa_codes_idx][:len(code_of_interest)]:
            code_present = True
        pa_codes_idx += 1
    return code_present

# TODO: move to helper file, may be used identically in other scripts
def old_long_format_PA(df_row, columns_retained, class_level=6, line_number=None, total_lines=None):
    """Function for changing the CPC code record of a single patent application 
        to long format with one row per CPC classification code.
       @param df_row (pandas DataFrame) record for this patent application
       @param columns_retained (list of str) which columns should be included
       @param class_level (int) up to what index should CPC classification be included 
                6 is maximum, 0 is minimum, 6 is default.
       @param line_number (int or None) number of the line in the dataframe being converted 
                used for debugging output
       @param total_lines (int or None) number of lines in the dataframe being converted 
                used for debugging output
       @return df (pandas DataFrame) data in long format
    """
    assert len(df_row) == 1
    assert "application_id" in columns_retained
    
    pa_codes = df_row['CPC_codes'].iloc[0]
    
    pa_codes = ["".join(pc[:class_level+1]) for pc in pa_codes]
    pa_codes = list(set(pa_codes))
    df_pas = pd.DataFrame({"Classification": pa_codes})
    df_pas["application_id"] = df_row["application_id"].iloc[0]
    
    df_row = df_row[columns_retained]
    df = pd.merge(df_row, df_pas, on="application_id", how="outer")
    
    if (line_number is not None) and (total_lines is not None) and ((line_number+1) % 1000 == 0):
        print("\rRow {0:d}/{1:d}".format(line_number+1, total_lines), end="")
    return df

# TODO: remove (not used here) or move to helper file, is used identically in other scripts
def old_urban_city_names(minimum_urban_city_size):
    """Function for obtaining a list of names of cities with population size above a threshold
       in various spellings
       @param minimum_urban_city_size (int) - size threshold
       
       @return (list of string) - Names
       """
    
    """ Load CSV file from here:
    https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/export/?disjunctive.cou_name_en&sort=name
    """
    dfurban = pd.read_csv("geonames-all-cities-with-a-population-1000.csv", sep=";")
    
    """ Explode df to account for every spelling"""
    dfurban["AltNames"] = dfurban.apply(lambda row: row["Alternate Names"].split(",") if isinstance(row["Alternate Names"], str) else [], axis=1)
    #dfurban[dfurban["ASCII Name"]=="Suwon"].iloc[0][["AltNames"]][0]
    dfurban2 = dfurban.explode("AltNames")
    dfurban2["Name"] = dfurban2["AltNames"]
    dfurban2 = dfurban.append(dfurban2)
    
    """ Filter by minimum size"""
    dfurban_redux = dfurban2[dfurban2["Population"] > minimum_urban_city_size]
    
    """ Extract list"""
    urban_names = list(dfurban_redux["Name"])
    
    return urban_names

# TODO: move to helper file, should be used identically in other scripts
def old_read_all_pa_to_df(do_reload=False, maximum_age=None, data_directory=DATA_DIR):
    """ Function for reading the data.
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param data_directory: Where count data are saved
        @return df (pandas DataFrame): DataFrame of patents 
    """
    
    data_filename = os.path.join(data_directory, "work", 'df_patents_list_short' + str(maximum_age) + '.pkl.gz')

    """ Check if data has to be reloaded"""
    if not do_reload:
        try:
            df = pd.read_pickle(data_filename, compression="gzip")
            print("Loading successful.")
            return df
        except:
            print("Loading has failed: ", sys.exc_info())
            do_reload = True
    
    """ Load and merge data"""
    filenames = get_processed_files()
    df = None
    for filename in filenames:
        #df_read = pd.read_pickle("data/processed/ipa210826.zip_patents.pkl.gz", compression="gzip")
        df_read = pd.read_pickle(filename, compression="gzip")
        current_file_total_size = sys.getsizeof(df_read)
        df_read.drop('description', axis=1, inplace=True)
        df_read.drop('claims', axis=1, inplace=True)
        current_file_net_size = sys.getsizeof(df_read)
        if (df is None):
            df = df_read.copy()
        else:
            df = df.append(df_read)
        df_total_size = sys.getsizeof(df)
        del df_read
        print("\rLoaded {0:s} of size {1:.2f} MB, reduced to {2:.2f} MB. Total df size now: {3:.2f} MB".format(
                            filename, current_file_total_size/2**20, current_file_net_size/2**20, df_total_size/2**20))
    
    """ Filter by age"""
    if maximum_age is not None:
        df["age_at_publication"] = df.apply(lambda row: (pd.to_datetime(row['publication_date']) - pd.to_datetime(row['application_date'])).days, axis=1)
        df = df[df["age_at_publication"] <= maximum_age]
    
    """ Save"""
    df.to_pickle(data_filename, compression="gzip")
    
    return df

# TODO: move to helper file, should be used identically in other scripts
def old_df_of_pa_by_classification(maximum_age=None, classification="subclass"):
    """ Function for reducing the patents data frame to ID, Date and one classification level.
        @param classification (str): Level at which Classification should be recorded
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @return df (pandas DataFrame with columns [ID, Classification, Date]): DataFrame of patents 
    """
    df = read_all_pa_to_df(maximum_age=maximum_age)
    cpc_levels = ["section", "class", "subclass", "main-group", "subgroup", "symbol-position", "classification-value"]
    cpc_class_level = cpc_levels.index(classification)
    total_lines = len(df)
    print("\nReducing CPC short format")
    dfs = [long_format_PA(df_row=df.iloc[[i]], 
                          columns_retained=["application_id", "application_date", 'country', 'state', 'city'],
                          class_level=cpc_class_level,
                          line_number=i,
                          total_lines=total_lines) for i in range(len(df))]
    print("\n")
    df = pd.concat(dfs, axis=0)
    assert (df.columns == ['application_id', 'application_date', 'country', 'state', 'city', 'Classification']).all()
    df.columns = ["ID", "Date", 'country', 'state', 'city', "Classification"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def count_patents_in_range(df, start=None, end=None):
    """ Function for counting patents in particular ranges in patent data frame
        @param df (pandas DataFrame with columns ["Classification", "Date", ...]: Patent Data
        @param start (pandas Datetime): start date of the counting period (inclusive)
        @param end (pandas Datetime): end date of the counting period (inclusive)
        @return df (pandas DataFrame with columns ["Classification", "Count", "Length"]): DataFrame
                    of patent classes and absolute frequency
    """
    
    """ Ascertain df is correct"""
    assert "Date" in df.columns
    assert "Classification" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    
    """ Subset columns"""
    df = df[["ID", "Date", "Classification"]]
    
    """ Fix start and end dates"""
    if start is None:
        start = min(df["Date"])
    if end is None:
        end = max(df["Date"])
    
    """ Discard rows outside of range"""
    df = df[(df["Date"]>=start) & (df["Date"]<=end)]
    
    """ Count patents by classification"""
    df_counts = df.groupby("Classification").agg("count").reset_index()
    if df["ID"].isnull().all():
        del df['ID']
    else:
        del df_counts["Date"]
    df_counts.columns = ["Classification", "Count"]

    """ Compute time length of count"""
    time_length = (end - start).days
    df_counts["Length"] = time_length
    
    return df_counts

def count_patents_in_range_weighted(df, start=None, end=None):
    """ Function for counting patents in particular ranges in patent data frame by classification, 
                but applying weights by the number of classification codes by patent
        @param df (pandas DataFrame with columns ["Classification", "Date", ...]: Patent Data
        @param start (pandas Datetime): start date of the counting period (inclusive)
        @param end (pandas Datetime): end date of the counting period (inclusive)
        @return df (pandas DataFrame with columns ["Classification", "Count", "Length"]): DataFrame
                    of patent classes and absolute frequency
    """
    
    """ Ascertain df is correct"""
    assert "Date" in df.columns
    assert "Classification_share" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    
    """ Subset columns"""
    df = df[["ID", "Date", "Classification", "Classification_share"]]
    
    """ Fix start and end dates"""
    if start is None:
        start = min(df["Date"])
    if end is None:
        end = max(df["Date"])
    
    """ Discard rows outside of range"""
    df = df[(df["Date"]>=start) & (df["Date"]<=end)]
    if df["ID"].isnull().all():
        del df['ID']
    
    """ Count patents by classification"""
    df_counts = df.groupby("Classification").agg("sum").reset_index()
    if 'Date' in df_counts.columns:
        del df_counts["Date"]
    df_counts.columns = ["Classification", "Count"]

    """ Compute time length of count"""
    time_length = (end - start).days
    df_counts["Length"] = time_length
    
    return df_counts


def visualize(df, 
              data_directory=DATA_DIR,
              output_directory=OUTPUT_DIR, 
              maximum_age=None, 
              longfilename_postfix=None, 
              axis_labels=None,
              x_column="Growth_period_x",
              y_column="Growth_period_y",
              emphasis_file=None,
              emphasis_number=50,
              plot_all_labels=False,
              limits=None,
              rectangle=None):
    """ Function for visualizing growth over 2 scales (so we can visualize in 2d)
        @param df (pandas DataFrame with columns ["Classification", "Total", 
                x_column, y_column]) - Data
        @param data_directory: Where count data are saved
        @param output_directory: Where plots and computed data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param longfilename_postfix (str or None): Output filename section for boundary dates.
        @param axis_labels (dict of str with keys "x", "y" or None): Axis labels        
        @param x_column (str) - column name of x-axis values
        @param y_column (str) - column name of y-axis values
        @param emphasis_file (str) - filename of file that contains the classification codes that 
                                     will be highlighted
        @param emphasis_number (int) - how many values to highlight
        @param plot_all_labels (bool) - Labels to be plotted for each value (not just highlighted)
        @param limits (dict numeric with keys xmin, xmax, ymin, ymax)
        @param rectangle (dict numeric with keys xmin, xmax, ymin, ymax)
    """
    
    
    """ Prepare axis labels"""
    try:
        x_label = axis_labels["x"]
        y_label = axis_labels["y"]
    except:
        x_label = "Long term growth rate"
        y_label = "Short term growth rate"
    
    """ Prepare plotting colors"""
    df["Mainsection"] = df["Classification"].str[:1]
    #e6194B1Green#3cb44b2Yellow#ffe1193Blue#4363d84Orange#f582315Purple#911eb46Cyan#42d4f47Magenta#f032e68Lime#bfef45
    df["col"] = df["Mainsection"].map({"A": "#e6194b", 
                                       "B": "#3cb44b",
                                       "C": "#4363d8", 
                                       "D": "#f58231",
                                       "E": "#911eb4", 
                                       "F": "#42d4f4",
                                       "G": "#f032e6", 
                                       "H": "#bfef45",
                                       "Y": "#000075"}) #469990, #dcbeff #9A6324
    df["MS_Name"] = df["Mainsection"].map({"A": "A - Human necessities", 
                                           "B": "B - Operations, transport",
                                           "C": "C - Chemistry, metallurgy", 
                                           "D": "D - Textiles, paper",
                                           "E": "E - Fixed constructions", 
                                           "F": "F - Engineering",
                                           "G": "G - Physics", 
                                           "H": "H - Electricity",
                                           "Y": "Y - New techn. tags"}) #469990, #dcbeff #9A6324
    
    df_labels = df[["col","MS_Name"]].drop_duplicates()
    
    """ Prepare plot marker sizes"""
    #df["size"] = np.log(df["Total"] + 1)
    df["size"] = np.log((df["Total"]/100 + 1)) * 100
    print(df)
    
    """ Prepare plot"""
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    
    """ Plot"""
    ax[0][0].scatter(df[x_column], df[y_column], s=df["size"], color=df["col"], alpha=0.5)
    if plot_all_labels:
        for i, row in df.iterrows():
            ax[0][0].annotate(row["Classification"], (row[x_column], row[y_column]), color=row["col"])

    """ Add emphasis"""
    if emphasis_file is not None:
        """ Load emphasis data"""
        df_emph = pd.read_pickle(os.path.join(data_directory, "work", emphasis_file), compression="gzip")
        df_emph = df_emph.iloc[:emphasis_number]
        df_emph = pd.merge(df, df_emph, on="Classification", how="inner")
        """ Plot emphasis"""
        ax[0][0].scatter(df_emph[x_column], df_emph[y_column], marker="x", s=100, linewidths=2, color="#000000")#, alpha=0.5)
        if not plot_all_labels:
            for i, row in df_emph.iterrows():
                ax[0][0].annotate("  "+row["Classification"], (row[x_column], row[y_column]), color=row["col"], )
            
        legend_handles = []
        for i, row in df_labels.iterrows():
            next_handle = mlines.Line2D([], [], color=row["col"], marker='.', linestyle='None', markersize=20, label=row["MS_Name"], alpha=0.5)
            legend_handles.append(next_handle)
        ax[0][0].legend(handles=legend_handles, loc='best')
    
    if limits is not None:
        ax[0][0].set_xlim(limits["xmin"], limits["xmax"])
        ax[0][0].set_ylim(limits["ymin"], limits["ymax"])
    
    if rectangle is not None:
        rect = patches.Rectangle((rectangle["xmin"], rectangle["ymin"]), 
                                  rectangle["xmax"]-rectangle["xmin"], 
                                  rectangle["ymax"]-rectangle["ymin"], 
                                  linewidth=1, edgecolor='r', facecolor='none')
        ax[0][0].add_patch(rect)
        ymin, ymax = ax[0][0].get_ylim()
        arr_width = 0.0015*(ymax-ymin)
        arr_start_x = rectangle["xmax"]
        arr_y = rectangle["ymin"] + (rectangle["ymax"]-rectangle["ymin"]) / 2
        xmin, xmax = ax[0][0].get_xlim()
        ax[0][0].set_xlim(xmin, xmax)
        arr_end_x = xmax  - 0.047 * (xmax-xmin) #+ (xmax-xmin)/30 
        ax[0][0].arrow(arr_start_x, arr_y, arr_end_x - arr_start_x, 0, width = arr_width, head_width = 13 * arr_width, 
                       head_length = 0.047 * (xmax-xmin), color='r')
        
    """ Fix axes"""
    ax[0][0].set_xlabel(x_label)
    ax[0][0].set_ylabel(y_label)
    xmin, xmax = ax[0][0].get_xlim()
    ymin, ymax = ax[0][0].get_ylim()
    ax[0][0].arrow(xmin, 0, xmax-xmin, 0, width=0, color="k")
    ax[0][0].arrow(0, ymin, 0, ymax-ymin, width=0, color="k")
    ax[0][0].plot([max(xmin,ymin), min(xmax,ymax)], [max(xmin,ymin), min(xmax,ymax)], color="k")
    ax[0][0].set_xlim(xmin, xmax)
    ax[0][0].set_ylim(ymin, ymax)
    
    """ Save plot"""
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "PA_all_subclasses_maximum_age_" + \
                    str(maximum_age) + ".pdf"))
    if longfilename_postfix is not None:
        plt.savefig(os.path.join(output_directory, "PA_all_subclasses_maximum_age_" + \
                    str(maximum_age) + longfilename_postfix + ".pdf"))
    """ Show plot"""
    #plt.show()


def multi_year_period_lists(period_start,
         period_end,
         accept_partial_years=False,
         key_prefix=""):
    """
    @param period_start
    @param period_end
    @param accept_partial_years (bool): 
    @param key_prefix (str):
    @return subperiod_dict (dict of dict of start: pd.TimeStamp, end: pd.TimeStamp, length: int)
    """
    
    import dateutil
    period = dateutil.relativedelta.relativedelta(period_end, period_start)
    
    try:
        assert period.years > 1
    except:
        assert (period.years == 1) and (period.months > 0), "Time length too small"
    
    if (period.months != 0) or (period.days != 0):
        if not accept_partial_years:
            assert False, "To compute an average annual growth reate, we need the period length to be full years."
        else:
            print("Partial years present, continuing anyway.")
    
    subperiod_dict = {}
    while (period_end - period_start).days > 0:
        subperiod = {"start": period_start, 
                     "end": period_start + pd.offsets.DateOffset(years=1),
                     "length": ((period_start + pd.offsets.DateOffset(years=1)) - period_start).days}
        subperiod_dict[key_prefix + str(period_start.year)] = subperiod
        period_start = period_start + pd.offsets.DateOffset(years=1)
    
    return subperiod_dict
    
        
def count_ranges_by_code(periods, 
         period_names={"period_x_before": "period_x_before", 
                       "period_x_after": "period_x_after", 
                       "period_y_before": "period_y_before", 
                       "period_y_after": "period_y_after"}, 
         do_reload=False,
         output_directory=OUTPUT_DIR, 
         data_directory=DATA_DIR,
         maximum_age=None,
         base_data_filename=None):
    """Function for counting numbers of patents per code (e.g., subclass, A12M, where A is the main 
                group, 12 the class, M the subclass)
        @param periods
        @param period_names
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param output_directory: Where plots and computed data are saved
        @param data_directory: Where count data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param base_data_filename (str): Filename of main input dataframe
        @return df_counts (pd dataframe): Dataframe of Classification codes and frequencies for periods
        @return longfilename_postfix (str): String of boundary dates to be used in output filenames 
    """
    
    """ Assert arguments are valid"""
    for period in period_names.values():
        assert period in periods.keys() 
    
    """ Prepare file names"""
    max_age_in_filename = str(maximum_age) if (maximum_age is not None) else "all"
    if base_data_filename is None:
        base_data_filename = "df_patents_subclass_dates_" + max_age_in_filename + ".pkl.gz"
    else: 
        base_data_save_filename = "df_patents_subclass_dates_" + max_age_in_filename + ".pkl.gz"
    base_data_filename = os.path.join(data_directory, "work", base_data_filename)
    count_data_filename = os.path.join(data_directory, 'work', "df_patents_by_subclass_" + \
                        max_age_in_filename + ".pkl.gz") 

    """ Load"""
    if not do_reload:
        try:
            #pdb.set_trace()
            df = pd.read_pickle(base_data_filename, compression="gzip")
            if len(df.columns) == 33 and (df.columns == [
                              'ID', 'Date', 'country', 'city', 
                              'state', 'publication_date', 'title_contains_Covid-19',
                              'abstract_contains_Covid-19', 'claims_contains_Covid-19',
                              'description_contains_Covid-19', 'title_contains_Covid19',
                              'abstract_contains_Covid19', 'claims_contains_Covid19',
                              'description_contains_Covid19', 'title_contains_2019-nCov',
                              'abstract_contains_2019-nCov', 'claims_contains_2019-nCov',
                              'description_contains_2019-nCov', 'title_contains_SARS-CoV-2',
                              'abstract_contains_SARS-CoV-2', 'claims_contains_SARS-CoV-2',
                              'description_contains_SARS-CoV-2',
                              'title_contains_pandemic_coronavirus',
                              'abstract_contains_pandemic_coronavirus',
                              'claims_contains_pandemic_coronavirus',
                              'description_contains_pandemic_coronavirus',
                              'title_contains_pandemic_covid', 'abstract_contains_pandemic_covid',
                              'claims_contains_pandemic_covid', 'description_contains_pandemic_covid',
                              'Classification', 'Classification_share', 'age_at_publication']).all():
                """This happens for base data input files generated by classes_by_terms_4.py, not by
                   basic_analysis_all_subclasses.py itself. In this case, we need to fix the columns and filter by age at publication."""
                if maximum_age is not None:
                    #pdb.set_trace()
                    #df["age_at_publication"] = df.apply(lambda row: (pd.to_datetime(row['publication_date']) - pd.to_datetime(row['Date'])).days, axis=1)
                    df = df[df["age_at_publication"] <= maximum_age]
                    #pd.to_pickle(base_data_filename[:-7] + "_reduced_maxage_" + str(maximum_age) + base_data_filename[-7:], compression="gzip")
                
                expected_columns = ['ID', 'Date', 'country', 'state', 'city', 'Classification', 'Classification_share', 'urban']
                """This one is missing"""
                #urban_names = analysis_helpers.urban_city_names(minimum_urban_city_size=100000)
                #df["urban"] = df.apply(lambda row: True if row["city"] in urban_names else False, axis=1)
                df["urban"] = None
                df = df[expected_columns]                                   

                """ Save"""
                df.to_pickle(base_data_save_filename, compression="gzip")
        except:
            print("Loading has failed: ", sys.exc_info())
            pdb.set_trace()
            do_reload = True
    if do_reload:
        """Reload"""
        df = analysis_helpers.df_of_pa_by_classification(maximum_age=maximum_age, classification="subclass")
        
        """ Add urbanness column"""
        urban_names = analysis_helpers.urban_city_names(minimum_urban_city_size=100000)
        df["urban"] = df.apply(lambda row: True if row["city"] in urban_names else False, axis=1)
        
        """ Save"""
        df.to_pickle(base_data_filename, compression="gzip")
        
    """ Obtain counts"""
    df_counts = count_patents_in_range_weighted(df, start=None, end=None)
    df_counts.columns = ["Classification", "Total", "Time_Length"]
    del df_counts["Time_Length"]
    
    longfilename_postfix = ""
    for period in periods.keys():
        #df_period_count = count_patents_in_range(df, start=periods[period]["start"], 
        #                                         end=periods[period]["end"])
        df_period_count = count_patents_in_range_weighted(df, start=periods[period]["start"], 
                                                 end=periods[period]["end"])
        df_period_count.columns = ["Classification", period+"_Count", period+"_Length"]
        df_counts = pd.merge(df_counts, df_period_count, on="Classification", how="outer")
        longfilename_postfix += "_" + str(periods[period]["start"].date()) + \
                                "_" + str(periods[period]["end"].date())
            
    """ Save"""
    df_counts.to_pickle(count_data_filename, compression="gzip")
    
    """ Return"""
    return df_counts, longfilename_postfix
    
def growth_rate_plot(periods, 
         period_names={"period_x_before": "period_x_before", 
                       "period_x_after": "period_x_after", 
                       "period_y_before": "period_y_before", 
                       "period_y_after": "period_y_after"}, 
         do_reload=False,
         output_directory=OUTPUT_DIR, 
         data_directory=DATA_DIR,
         maximum_age=None,
         classes_excluded_from_figure=[],
         separate_by_maingroups=False,
         axis_labels=None,
         emphasis_file=None,
         filename_addition=""):
    """Function for plotting two growth rates of patent code frequencies against one another (i.e., for 
            2 growth rates, each from one time range to another). Intended for comparing pre-pandemic and 
            in-pandemic growth.
        @param periods
        @param period_names
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param output_directory: Where plots and computed data are saved
        @param data_directory: Where count data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param classes_excluded_from_figure (list of str): List of classes that should be excluded from 
                                    visualization (for outliers, to make the body of the distribution visible)
        @param separate_by_maingroups (bool): Should additional visualization separate for each maingroup 
                                                be added?
        @param axis_labels (dict of str with keys "x", "y" or None): Axis labels
        @param emphasis_file (str) - filename of file that contains the classification codes that 
                                     will be highlighted
        @param filename_addition (str) - string to be inserted in output filenames for those to be 
                                         distinguishable
    """
    
    """ Obtain data"""
    df_counts, longfilename_postfix = count_ranges_by_code(periods, 
                                            period_names=period_names, 
                                            do_reload=do_reload,
                                            maximum_age=maximum_age)
        
    """ Obtain growth rates"""
    for key in ["period_x", "period_y"]:
        key_before_count = period_names[key + "_before"]+"_Count"
        key_before_length = period_names[key + "_before"]+"_Length"
        key_after_count = period_names[key + "_after"]+"_Count"
        key_after_length = period_names[key + "_after"]+"_Length"
        df_counts["Growth_"+key] = (df_counts[key_after_count] / df_counts[key_after_length] - \
                                   df_counts[key_before_count] / df_counts[key_before_length]) / \
                                   (df_counts[key_before_count] / df_counts[key_before_length]) 
    
    print(df_counts)
    #pdb.set_trace()
    
    """ Visualize"""
    #df_counts = df_counts[~df_counts['Classification'].isin(['G17Y', "L17Y"])]
    df_counts = df_counts[~df_counts['Classification'].isin(classes_excluded_from_figure)]
    visualize(df_counts[["Classification", "Total", "Growth_period_x", "Growth_period_y"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix=filename_addition+"_growth_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Growth_period_x",
              y_column="Growth_period_y",
              emphasis_file=emphasis_file)
    
    if separate_by_maingroups:
        for maingroup in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
            df_counts_subset = df_counts[df_counts["Classification"].str[:1] == maingroup]
            visualize(df_counts_subset[["Classification", "Total", "Growth_period_x", "Growth_period_y"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix=filename_addition+"_growth_"+maingroup+"_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Growth_period_x",
              y_column="Growth_period_y",
              emphasis_file=emphasis_file)
    

def avg_growth_rate_plot(periods, 
         do_reload=False,
         output_directory=OUTPUT_DIR, 
         data_directory=DATA_DIR,
         maximum_age=None,
         classes_excluded_from_figure=[],
         separate_by_maingroups=False,
         axis_labels=None,
         emphasis_file=None,
         emphasis_number=50,
         filename_addition="",
         plot_all_labels=False,
         limits=None,
         rectangle=None,
         base_data_filename=None): #period_names={"period_x": "period_x", "period_y": "period_y"}, 
    """Function for plotting two growth rates of patent code frequencies against one another (i.e., for 
            2 growth rates, each from one time range to another). Intended for comparing pre-pandemic and 
            in-pandemic growth.
        @param periods
        #@param period_names
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param output_directory: Where plots and computed data are saved
        @param data_directory: Where count data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param classes_excluded_from_figure (list of str): List of classes that should be excluded from 
                                    visualization (for outliers, to make the body of the distribution visible)
        @param separate_by_maingroups (bool): Should additional visualization separate for each maingroup 
                                                be added?
        @param axis_labels (dict of str with keys "x", "y" or None): Axis labels
        @param emphasis_file (str) - filename of file that contains the classification codes that 
                                     will be highlighted
        @param emphasis_number (int) - how many values to highlight
        @param filename_addition (str) - string to be inserted in output filenames for those to be 
        @param filename_addition (str) - string to be inserted in output filenames for those to be 
                                         distinguishable
        @param plot_all_labels (bool)
        @param limits (dict numeric with keys xmin, xmax, ymin, ymax)
        @param rectangle (dict numeric with keys xmin, xmax, ymin, ymax)
        @param base_data_filename (str): Filename of main input dataframe
    """
    
    """ Define individual year time periods"""
    joined_period_dict = {}
    key_lists = {"period_x": [], "period_y": []}
    for key in ["period_x", "period_y"]:
        #period_name = period_names[key]
        period_dict = multi_year_period_lists(key_prefix=key + "_",
                                                period_start=periods[key]["start"], 
                                                period_end=periods[key]["end"])
        """ Avoid duplicating values"""
        for key2, period in period_dict.items():
            pkey = key2
            if period in joined_period_dict.values():
                """ Find key by value"""
                pkey = list(joined_period_dict.keys())[list(joined_period_dict.values()).index(period)]
            else:
                joined_period_dict[key2] = period
            key_lists[key].append(pkey)
        
        
    """ Obtain data"""
    period_names = {k: k for k in joined_period_dict.keys()}
    df_counts, _ = count_ranges_by_code(periods = joined_period_dict, 
                                        period_names=period_names, 
                                        do_reload=do_reload,
                                        maximum_age=maximum_age,
                                        base_data_filename=base_data_filename)
    
    """ Compute average growth rates"""
    for key in ["period_x", "period_y"]:
        period_keys = key_lists[key]
        df_counts["Growth_" + key] = 0
        for pkey in period_keys:
            """ Catch partial years and throw error since this is not implemented"""
            assert ((df_counts[pkey + "_Length"] >= 364) | pd.isna(df_counts[pkey + "_Length"])).all(), "Partial year " + pkey
            if "Count_Old" in df_counts:
                df_counts["Growth_" + key] = df_counts["Growth_" + key] + \
                                      (df_counts[pkey + "_Count"] - df_counts["Count_Old"]) /\
                                                          df_counts["Count_Old"]
            df_counts["Count_Old"] = df_counts[pkey + "_Count"]
        df_counts["Growth_" + key] = df_counts["Growth_" + key] / (len(period_keys) - 1)
        del df_counts["Count_Old"]
    
    """ Create filename postfix"""
    longfilename_postfix = "_average_GR" 
    for key in ["period_x", "period_y"]:
        longfilename_postfix += "_" + str(periods[key]["start"].date()) +\
                                "_" + str(periods[key]["end"].date())
    
    
    """ Visualize"""
    #df_counts = df_counts[~df_counts['Classification'].isin(['G17Y', "L17Y"])]
    df_counts = df_counts[~df_counts['Classification'].isin(classes_excluded_from_figure)]
    #pdb.set_trace()
    visualize(df_counts[["Classification", "Total", "Growth_period_x", "Growth_period_y"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix=filename_addition+"_growth_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Growth_period_x",
              y_column="Growth_period_y",
              emphasis_file=emphasis_file,
              emphasis_number=emphasis_number,
              plot_all_labels=plot_all_labels,
              limits=limits,
              rectangle=rectangle)
    
    if separate_by_maingroups:
        for maingroup in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
            df_counts_subset = df_counts[df_counts["Classification"].str[:1] == maingroup]
            visualize(df_counts_subset[["Classification", "Total", "Growth_period_x", "Growth_period_y"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix=filename_addition+"_growth_"+maingroup+"_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Growth_period_x",
              y_column="Growth_period_y",
              emphasis_file=emphasis_file,
              emphasis_number=emphasis_number,
              plot_all_labels=plot_all_labels)
    


def count_plot(periods, 
         period_names={"Period_x": "Period_x", 
                       "Period_y": "Period_y"}, 
         do_reload=False,
         output_directory=OUTPUT_DIR, 
         data_directory=DATA_DIR,
         maximum_age=None,
         classes_excluded_from_figure=[],
         separate_by_maingroups=False,
         axis_labels=None,
         emphasis_file=None,
         filename_addition=""):
    """Function for plotting absolute frequencies of patents per code string (e.g. subclass, A12M, 
                where A is the main group, 12 the class, M the subclass) for two time ranges against
                one another. Intended for comparing pre-pandemic and in-pandemic frequencies.
        @param periods
        @param period_names
        @param do_reload (bool): Should DataFrame be recreated even if it already exists on disk
        @param output_directory: Where plots and computed data are saved
        @param data_directory: Where count data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
        @param classes_excluded_from_figure (list of str): List of classes that should be excluded from 
                                    visualization (for outliers, to make the body of the distribution visible)
        @param separate_by_maingroups (bool): Should additional visualization separate for each maingroup 
                                                be added?
        @param axis_labels (dict of str with keys "x", "y" or None): Axis labels
        @param emphasis_file (str) - filename of file that contains the classification codes that 
                                     will be highlighted
        @param filename_addition (str) - string to be inserted in output filenames for those to be 
                                         distinguishable
    """
    
    """ Obtain data"""
    df_counts, longfilename_postfix = count_ranges_by_code(periods, 
                                            period_names=period_names, 
                                            do_reload=do_reload,
                                            maximum_age=maximum_age)
        
    print(df_counts)
    #pdb.set_trace()
    
    """ Visualize"""
    #df_counts = df_counts[~df_counts['Classification'].isin(['G17Y', "L17Y"])]
    df_counts = df_counts[~df_counts['Classification'].isin(classes_excluded_from_figure)]
    visualize(df_counts[["Classification", "Total", "Period_x_Count", "Period_y_Count"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix=filename_addition+"_count_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Period_x_Count",
              y_column="Period_y_Count",
              emphasis_file=emphasis_file)
    
    if separate_by_maingroups:
        for maingroup in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
            df_counts_subset = df_counts[df_counts["Classification"].str[:1] == maingroup]
            visualize(df_counts_subset[["Classification", "Total", "Period_x_Count", "Period_y_Count"]], 
              output_directory=output_directory, 
              maximum_age=maximum_age,
              longfilename_postfix="_count_"+maingroup+"_"+longfilename_postfix,
              axis_labels=axis_labels,
              x_column="Period_x_Count",
              y_column="Period_y_Count",
              emphasis_file=emphasis_file)

def main_old(output_directory=OUTPUT_DIR, maximum_age=None):
    """Function...
        @param output_directory: Where plots and computed data are saved
        @param maximum_age (int or None): maximum age beyond which patent applications are filtered out. 
                                    This is for making P.A.s filed at different times comparable
    """
    
    """ Load and merge data"""
    filenames = get_processed_files()
    df = None
    for filename in filenames:
        #df_read = pd.read_pickle("data/processed/ipa210826.zip_patents.pkl.gz", compression="gzip")
        df_read = pd.read_pickle(filename, compression="gzip")
        current_file_total_size = sys.getsizeof(df_read)
        df_read.drop('description', axis=1, inplace=True)
        df_read.drop('claims', axis=1, inplace=True)
        current_file_net_size = sys.getsizeof(df_read)
        if (df is None):
            df = df_read.copy()
        else:
            df = df.append(df_read)
        df_total_size = sys.getsizeof(df)
        del df_read
        print("\rLoaded {0:s} of size {1:.2f} MB, reduced to {2:.2f} MB. Total df size now: {3:.2f} MB".format(
                            filename, current_file_total_size/2**20, current_file_net_size/2**20, df_total_size/2**20))
    
    """ Filter by age"""
    if maximum_age is not None:
        df["age_at_publication"] = df.apply(lambda row: (pd.to_datetime(row['publication_date']) - pd.to_datetime(row['application_date'])).days, axis=1)
        df = df[df["age_at_publication"] <= maximum_age]
    
    
    #pdb.set_trace()    
    #ids = df["application_id"]
    #duplicates = df[ids.isin(ids[ids.duplicated()])].sort_values("application_id")
    
    """ Identify urban and rural applicant locations"""
    urban_names = urban_city_names(minimum_urban_city_size=100000)
    df["urban"] = df.apply(lambda row: True if row["city"] in urban_names else False, axis=1)
    df["rural"] = ~df["urban"]
    urbanness_columns = ["urban", "rural"]
    
    """ Identify countries"""
    country_IDs = ["US", "DE", "JP", "CN", "IT", "FR", "KR", "TW", "GB", "CA", "CH", "NL", "SE", "IL"]
    country_columns = ["Country_" + cid for cid in country_IDs]
    for i, cid in enumerate(country_IDs):
        df[country_columns[i]] = df["country"] == cid
    
    """ Identify main groups """
    """for each CPC code idx 0 is main group. E.g.: 
        [['A', '01', 'B', '63', '1006', 'F', 'I'], 
         ['A', '01', 'B', '76', '00', 'L', 'I'], 
         ['A', '01', 'B', '63', '118', 'L', 'I'], 
         ['A', '01', 'B', '59', '067', 'L', 'I']]"""
    df['CPC_MGs'] = df.apply(lambda row: [CC[0] for CC in row["CPC_codes"]], axis=1)
    for MG in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        df['CPC_MG_' + MG] = df.apply(lambda row: True if MG in row["CPC_MGs"] else False, axis=1)
    
    codes_of_interest = [["A", "61", "K"], # Medicinal preparations 
                         ["C", "12", "M"], # Apparatuses for enzymology
                         ["C", "12", "N"], # Microorganisms 
                         ["C", "12", "Q"], # Measuring or testing processes involving enzymes, nucleic acids, microorganisms
                         ["H", "04", "L"], # Transmission of digital information, telegraphics
                         ["H", "04", "M"], # Telephonic communication
                         ["H", "04", "N"], # Pictoral communication
                         ["C", "12", "N", "7"],         # Virusses, bacteriophages
                         ["C", "12", "N", "15"],        # DNA or RNA concerning genetic engineering, vectors
                         ["A", "61", "K", "39"],        # Medicinal preparations containing antigens or antibodies
                         ["A", "61", "K", "39", "12"],  # Viral antigens
                         ["A", "61", "K", "39", "215"], # Medicinal preparations containing antigens or antibodies: Coronaviridae
                         ["C", "12", "N", "2770"],      # ssRNA Viruses positive-sense
                         ["H", "04", "N", "7", "15"]]   # Video conference systems
    
    codes_of_interest_column_names = ["COI_" + "".join(coi) for coi in codes_of_interest]                     
    for coi_idx, coi in enumerate(codes_of_interest):
        df[codes_of_interest_column_names[coi_idx]] = df.apply(lambda row: find_code_in_PA(pa_codes=row["CPC_codes"], 
                                                                                           code_of_interest=coi), axis=1)

    """ Create reduced dataframe with only the columns we want (in this case the category indicators)"""
    keep_columns = ["application_date"]
    for MG in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        keep_columns.append('CPC_MG_' + MG)
        #df1 = df[['CPC_MG_' + MG]]
    keep_columns += codes_of_interest_column_names + urbanness_columns + country_columns
    df1 = df[keep_columns]
    df1["all"] = np.ones(len(df1))

    """ Count number in each category"""
    df_counts = df1.groupby(["application_date"]).sum()
    #df_counts["all"] = df1.groupby(["application_date"]).agg("count")

    """ Create date variables for year and day of year"""
    df_counts = df_counts.sort_index()
    df_counts["Date"] = pd.to_datetime(df_counts.index)
    df_counts["Year"] = df_counts.apply(lambda row: row["Date"].year, axis=1)
    df_counts["Day"] = df_counts.apply(lambda row: (row["Date"] - pd.to_datetime(row["Date"].year, format="%Y")).days + 1, axis=1)
    #print(list(df_counts["Day"].value_counts().sort_index()))

    """ Plotting (for each category)"""
    df_plots = {}
    plot_columns = ['all', 'CPC_MG_A', 'CPC_MG_B', 'CPC_MG_C', 'CPC_MG_D', 'CPC_MG_E', 'CPC_MG_F', 
                                                'CPC_MG_G', 'CPC_MG_H', 'CPC_MG_Y']
    plot_columns += codes_of_interest_column_names + urbanness_columns + country_columns
    
    for plot_column in plot_columns:
        """ Apply MA filter before pivot to wide format"""
        df_counts[plot_column + '_MA'] = df_counts[plot_column].rolling(window=7).mean()
        
        """ Transform to wide format for current variable (day of year vs year)"""
        df_plot = df_counts.pivot(index='Day', columns='Year', values=plot_column+'_MA')
        
        """ Plot"""
        create_plot(df_plot, plot_column, output_directory, maximum_age)
        
        """ Save plotting data"""
        df_plots[plot_column] = df_plot
        
        """ Restore original df_counts"""
        df_counts.drop(plot_column + '_MA', axis=1, inplace=True)
    
    """ Save data (df_counts, df_plots, df)"""
    df.to_pickle(os.path.join(output_directory, 'df_patent_applications_merged_wo_description_maximum_age_' + str(maximum_age) + '.pkl.gz'), compression="gzip")
    df_counts.to_pickle(os.path.join(output_directory, 'df_number_patent_applications_by_MG_maximum_age_' + str(maximum_age) + '.pkl.gz'), compression="gzip")
    with gzip.open(os.path.join(output_directory, 'patent_applications_plotting_dfs_as_list_maximum_age_' + str(maximum_age) + '.pkl.gz'), 'wb') as wfile:
        pickle.dump(df_plots, wfile)

    """ Check that clean reload of saved data is possible"""
    df = pd.read_pickle(os.path.join(output_directory, 'df_patent_applications_merged_wo_description_maximum_age_' + str(maximum_age) + '.pkl.gz'), compression="gzip")
    df_counts = pd.read_pickle(os.path.join(output_directory, 'df_number_patent_applications_by_MG_maximum_age_' + str(maximum_age) + '.pkl.gz'), compression="gzip")
    with gzip.open(os.path.join(output_directory, 'patent_applications_plotting_dfs_as_list_maximum_age_' + str(maximum_age) + '.pkl.gz'), "rb") as rfile:
        df_plots = pickle.load(rfile)

def main():
    """Main function
    """
    if True:
        base_data_filename = "df_Covid-19_all_terms_subclasses.pkl.gz"
        #base_data_filename = None
        #ef = "df_Covid-19_subclasses_shares.pkl.gz"
        #ef = "df_Covid-19_all_terms_subclasses_shares_fulltext_weighted.pkl.gz"
        #ef = "df_Covid-19_terms_subclasses_shares_fulltext_weighted.pkl.gz"
        ef = "df_Covid-19_terms_subclasses_shares_fulltext_weighted_reduced.pkl.gz"
        # main analysis
        print("\n\n\n-------------\nMain analysis\n-------------\n\n\n")
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    #classes_excluded_from_figure=["B41P", "Y02T", "D06L", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "G06G", "Y02E"],#"D06L", "G06E", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "F05B", "G06G", "Y02E" #"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2019-2020 (pandemic)", "x": "Average annual growth 2015-2019 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_restricted_emph_43_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename,
                                    limits={'xmin':-0.2, 'xmax':1.65, 'ymin': -0.82, 'ymax': 1.08}
                                    )
        # rectangle sequence
                # main analysis
        print("\n\n\n-------------\nRectangle sequence\n-------------\n\n\n")
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=False,
                                    axis_labels={"y": "Growth 2019-2020 (pandemic)", "x": "Average annual growth 2015-2019 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_RECTANGLE0_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename,
                                    rectangle={'xmin':-0.2, 'xmax':1.65, 'ymin': -0.82, 'ymax': 1.08}
                                    )
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=False,
                                    axis_labels={"y": "Growth 2019-2020 (pandemic)", "x": "Average annual growth 2015-2019 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_RECTANGLE1_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename,
                                    rectangle={'xmin':-0.1, 'xmax':0.7, 'ymin':-0.5, 'ymax':0.47},
                                    limits={'xmin':-0.2, 'xmax':1.65, 'ymin': -0.82, 'ymax': 1.08}
                                    )
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=False,
                                    axis_labels={"y": "Growth 2019-2020 (pandemic)", "x": "Average annual growth 2015-2019 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_RECTANGLE2_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename,
                                    limits={'xmin':-0.1, 'xmax':0.7, 'ymin':-0.5, 'ymax':0.47}
                                    )

        # validation test
        print("\n\n\n-------------\nValidation test\n-------------\n\n\n")
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2018-04-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x": {'start': pd.to_datetime("2014-04-01"), 'end': pd.to_datetime("2019-04-01")}},
                                    classes_excluded_from_figure=["C08C", "Y02T", "G09C", "B60Y", "E05Y", "C14B", "B61J", "B60V", "D01B", "F16S", "G06G", "F05B", "Y02B", "Y04S", "Y02E", "A44D"],#"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2018-2019", "x": "Average annual growth 2014-2018"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_restricted_emph_43_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename)
        print("\n\n\n-------------\nValidation test\n-------------\n\n\n")
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2017-04-01"), 'end': pd.to_datetime("2019-04-01")},
                                   "period_x": {'start': pd.to_datetime("2014-04-01"), 'end': pd.to_datetime("2018-04-01")}},
                                    classes_excluded_from_figure=["C08C", "Y02T", "G09C", "B60Y", "E05Y", "C14B", "B61J", "B60V", "D01B", "F16S", "G06G", "F05B", "Y02B", "Y04S", "Y02E", "A44D"],#"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2017-2018", "x": "Average annual growth 2014-2017"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_restricted_emph_43_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename)
        # upstream subclasses
        # TODO: Replace
        ef = "df_upstream_from_Covid-19_abs_50.pkl.gz"
        print("\n\n\n-------------\nUpstream subclasses\n-------------\n\n\n")
        avg_growth_rate_plot(maximum_age=555, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=["B41P", "Y02T", "D06L", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "G06G", "Y02E"],#, "H03C", "G10F" #"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2019-2020 (pandemic)", "x": "Average annual growth 2015-2019 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=43,
                                    filename_addition="_restricted_emph_upstream_43_",
                                    plot_all_labels=False,
                                    base_data_filename=base_data_filename,
                                    limits={'xmin':-0.2, 'xmax':1.65, 'ymin': -0.82, 'ymax': 1.08}
                                    )

    if False:
        """548 Days (1.5 years) - First 6 months vs same time 2019"""
        #for ef in ["df_Covid-19_subclasses_counts.pkl.gz", "df_Covid-19_subclasses_shares.pkl.gz"]:
        count_plot(maximum_age=548, periods={"Period_x": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "Period_y": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True,
                                    #axis_labels={"x": "Pandemic trend (2015-2020 vs. 2020-21)", "y": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "# 04/2019-09/2019", "y": "#04/2020-09/2020"},
                                    emphasis_file=ef)
        """Comparing counts before and in pandemic // 04/2019-03/2020 vs first year of pandemic (04/2020-03/2021)"""
        count_plot(maximum_age=366, periods={"Period_x": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "Period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True,
                                    #axis_labels={"y": "Pandemic trend (2015-2020 vs. 2020-21)", "x": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "# 04/2019-03/2020", "y": "#04/2020-03/2021"},
                                    emphasis_file=ef)
        """First 6 months vs second 6 months"""
        count_plot(maximum_age=366, periods={"Period_x": {'start': pd.to_datetime("2020-10-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "Period_y": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=[], #G16Y, D06L
                                    separate_by_maingroups=True,
                                    axis_labels={"x": "Number Oct-Mar", "y": "Number Growth Apr-Sep"},
                                    emphasis_file=ef)
    
    ef = "df_Covid-19_subclasses_shares.pkl.gz"
    
    if False:
        """548 Days (1.5 years) - First 6 months vs same time 2019, vs 6 months 2019 vs same time 2018"""
        #for ef in ["df_Covid-19_subclasses_counts.pkl.gz", "df_Covid-19_subclasses_shares.pkl.gz"]:
        growth_rate_plot(maximum_age=548, periods={"period_x_before": {'start': pd.to_datetime("2018-04-01"), 'end': pd.to_datetime("2018-10-01")},
                               "period_x_after": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                               "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                               "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                classes_excluded_from_figure=["C12J", "G16B", "G16C"],#C12J, G1gB, G16C
                                separate_by_maingroups=True,
                                #axis_labels={"x": "Pandemic trend (2015-2020 vs. 2020-21)", "y": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                axis_labels={"x": "Growth 2018-2019", "y": "Growth 2019-2020"},
                                emphasis_file=ef)
        """Comparing trends before and in pandemic // First year (2020-21) vs 2016-2020, vs growth from 2015-2019 to 2019-20"""
        growth_rate_plot(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2016-04-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2019-04-01")},
                                   "period_y_after": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True,
                                    #axis_labels={"y": "Pandemic trend (2015-2020 vs. 2020-21)", "x": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "Pandemic trend", "y": "Pre-pandemic trend"},
                                    emphasis_file=ef)
        growth_rate_plot(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2016-04-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2019-04-01")},
                                   "period_y_after": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=["D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=False,
                                    #axis_labels={"y": "Pandemic trend (2015-2020 vs. 2020-21)", "x": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "Pandemic trend", "y": "Pre-pandemic trend"},
                                    emphasis_file=ef,
                                    filename_addition="_restricted_")
        """First 6 months vs same time 2019, vs second 6 months vs same time 2019"""
        growth_rate_plot(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2019-10-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-10-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=["G16Y", "D06L"], #G16Y, D06L
                                    separate_by_maingroups=True,
                                    axis_labels={"x": "Annual growth Oct-Mar", "y": "Annual Growth Apr-Sep"},
                                    emphasis_file=ef)
        """548 Days (1.5 years) - First 6 months vs same time 2019, vs 6 months 2019 vs same time 2018"""
        #for ef in ["df_Covid-19_subclasses_counts.pkl.gz", "df_Covid-19_subclasses_shares.pkl.gz"]:
        avg_growth_rate_plot(maximum_age=548, periods={"period_x": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],#C12J, G1gB, G16C
                                    separate_by_maingroups=True,
                                    axis_labels={"x": "Pandemic trend", "y": "Pre-pandemic trend"},
                                    emphasis_file=ef,
                                    plot_all_labels=True)
                                            
        #ef = "df_Covid-19_subclasses_shares_fulltext.pkl.gz"
        avg_growth_rate_plot(maximum_age=366, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],#"D06L", "G06E", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "F05B", "G06G", "Y02E"
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2020-2021 (pandemic)", "x": "Average annual growth 2015-2020 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=50,
                                    plot_all_labels=False)
    #if True:
    if False:
        avg_growth_rate_plot(maximum_age=366, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=["D06L", "G06E", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "F05B", "G06G", "Y02E"],#"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2020-2021 (pandemic)", "x": "Average annual growth 2015-2020 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=50,
                                    filename_addition="_restricted_",
                                    plot_all_labels=False)
        avg_growth_rate_plot(maximum_age=366, periods={"period_y": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_x": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=["D06L", "G06E", "Y02P", "Y10T", "Y02B", "Y04S", "C10N", "A44D", "F05B", "G06G", "Y02E"],#"D06G", "G06J", "G16B", "G16C"],
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Growth 2020-2021 (pandemic)", "x": "Average annual growth 2015-2020 (pre-pandemic)"},
                                    emphasis_file=ef,
                                    emphasis_number=20,
                                    filename_addition="_restricted_emph_20_",
                                    plot_all_labels=False)

if __name__ == '__main__':
    main()
    
    if False:
        """Years vs Months"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2019-01-01"), 'end': pd.to_datetime("2020-01-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-01-01"), 'end': pd.to_datetime("2021-01-01")},
                                   "period_y_before": {'start': pd.to_datetime("2020-02-01"), 'end': pd.to_datetime("2020-03-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-05-01")}},
                                    classes_excluded_from_figure=["G16Y"])
        """First half year vs everything after"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2015-01-01"), 'end': pd.to_datetime("2020-03-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-03-01"), 'end': pd.to_datetime("2022-09-01")},
                                   "period_y_before": {'start': pd.to_datetime("2015-01-01"), 'end': pd.to_datetime("2020-03-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-03-01"), 'end': pd.to_datetime("2020-08-01")}},
                                    classes_excluded_from_figure=["G16Y", "G06J", "G16Z", "B68F", "B62C"])
        """First month vs first 6 months compared to same time 2019"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")},
                                   "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-05-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-05-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True)
        """First 6 months vs same time 2019, vs 6 months vs same time 2015"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2015-10-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")},
                                   "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True)
        #if True:
        """548 Days (1.5 years) - First 6 months vs same time 2019, vs 6 months vs same time 2015"""
        main(maximum_age=548, periods={"period_x_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2015-10-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")},
                                   "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True,
                                    #axis_labels={"x": "Pandemic trend (2015-2020 vs. 2020-21)", "y": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "Growth 2015-2020", "y": "Growth 2019-2020"})
        """Comparing trends before and in pandemic // First year (2020-21) vs 2015-2020, vs growth from 2015-2019 to 2019-20"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y_before": {'start': pd.to_datetime("2015-04-01"), 'end': pd.to_datetime("2019-04-01")},
                                   "period_y_after": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2020-04-01")}},
                                    classes_excluded_from_figure=[],
                                    separate_by_maingroups=True,
                                    #axis_labels={"y": "Pandemic trend (2015-2020 vs. 2020-21)", "x": "Pre-pandemic trend (2015-19 vs 2019-20)"})
                                    axis_labels={"x": "Pandemic trend", "y": "Pre-pandemic trend"})
        """First 6 months vs same time 2019, vs second 6 months vs same time 2019"""
        main(maximum_age=366, periods={"period_x_before": {'start': pd.to_datetime("2019-10-01"), 'end': pd.to_datetime("2020-04-01")},
                                   "period_x_after": {'start': pd.to_datetime("2020-10-01"), 'end': pd.to_datetime("2021-04-01")},
                                   "period_y_before": {'start': pd.to_datetime("2019-04-01"), 'end': pd.to_datetime("2019-10-01")},
                                   "period_y_after": {'start': pd.to_datetime("2020-04-01"), 'end': pd.to_datetime("2020-10-01")}},
                                    classes_excluded_from_figure=[], #G16Y, D06L
                                    separate_by_maingroups=True,
                                    axis_labels={"y": "Annual growth Oct-Mar", "x": "Annual Growth Apr-Sep"})
        
