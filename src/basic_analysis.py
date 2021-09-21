import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import gzip
import pickle
import glob
import os
import pdb

from settings import DATA_DIR, OUTPUT_DIR

def get_processed_files(directory=os.path.join(DATA_DIR, 'processed')):
    """Function for collecting filenames of already downloaded raw files.
    @param directory: place where the files are stored
    @return: list of zip files
    """
    return glob.glob(os.path.join(directory, '*.zip_patents.pkl.gz'))

def create_plot(df_plot, plot_column, output_directory):
	""" Function for creating plot of the number of patent applications in a certain category 
	    (e.g. all or certain main groups).
	    @param df_plot: pandas dataframe with the data in wide format (Day of the year vs Year)
	    @param plot_column: Category that is currently being plotted
	    @param output_directory: Where plots are saved
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
	plt.savefig(os.path.join(output_directory, "patent_applications_" + plot_column + ".pdf"))
	#plt.show()


def main(output_directory=OUTPUT_DIR):
	"""Function...
	    @param output_directory: Where plots and computed data are saved
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
	
	#pdb.set_trace()	
	#ids = df["application_id"]
	#duplicates = df[ids.isin(ids[ids.duplicated()])].sort_values("application_id")
	
	""" Identify main groups """
	"""for each CPC code idx 0 is main group. E.g.: 
		[['A', '01', 'B', '63', '1006', 'F', 'I'], 
		 ['A', '01', 'B', '76', '00', 'L', 'I'], 
		 ['A', '01', 'B', '63', '118', 'L', 'I'], 
		 ['A', '01', 'B', '59', '067', 'L', 'I']]"""
	df['CPC_MGs'] = df.apply(lambda row: [CC[0] for CC in row["CPC_codes"]], axis=1)
	for MG in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
		df['CPC_MG_' + MG] = df.apply(lambda row: True if MG in row["CPC_MGs"] else False, axis=1)

	""" Create reduced dataframe with only the columns we want (in this case the category indicators)"""
	keep_columns = ["application_date"]
	for MG in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
		keep_columns.append('CPC_MG_' + MG)
		#df1 = df[['CPC_MG_' + MG]]
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
	for plot_column in ['all', 'CPC_MG_A', 'CPC_MG_B', 'CPC_MG_C', 'CPC_MG_D', 'CPC_MG_E', 'CPC_MG_F', 
																	'CPC_MG_G', 'CPC_MG_H', 'CPC_MG_Y']:
		""" Apply MA filter before pivot to wide format"""
		df_counts[plot_column + '_MA'] = df_counts[plot_column].rolling(window=7).mean()
		
		""" Transform to wide format for current variable (day of year vs year)"""
		df_plot = df_counts.pivot(index='Day', columns='Year', values=plot_column+'_MA')
		
		""" Plot"""
		create_plot(df_plot, plot_column, output_directory)
		
		""" Save plotting data"""
		df_plots[plot_column] = df_plot
		
		""" Restore original df_counts"""
		df_counts.drop(plot_column + '_MA', axis=1, inplace=True)
	
	""" Save data (df_counts, df_plots, df)"""
	df.to_pickle(os.path.join(output_directory, "df_patent_applications_merged_wo_description.pkl.gz"), compression="gzip")
	df_counts.to_pickle(os.path.join(output_directory, "df_number_patent_applications_by_MG.pkl.gz"), compression="gzip")
	with gzip.open(os.path.join(output_directory, 'patent_applications_plotting_dfs_as_list.pkl.gz'), 'wb') as wfile:
		pickle.dump(df_plots, wfile)

	""" Check that clean reload of saved data is possible"""
	df = pd.read_pickle(os.path.join(output_directory, "df_patent_applications_merged_wo_description.pkl.gz"), compression="gzip")
	df_counts = pd.read_pickle(os.path.join(output_directory, "df_number_patent_applications_by_MG.pkl.gz"), compression="gzip")
	with gzip.open(os.path.join(output_directory, 'patent_applications_plotting_dfs_as_list.pkl.gz'), "rb") as rfile:
		df_plots = pickle.load(rfile)

if __name__ == '__main__':
    main()
