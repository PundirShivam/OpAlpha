import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from copy import deepcopy


def read_custom_csv(csv_path):
	#R"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\spx_eod_201801.txt"
	df = pd.read_csv(csv_path,delimiter=", ")
	df.columns = df.columns.str.replace("[","").str.replace("]","")

	df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'],format='%Y-%m-%d')
	df['QUOTE_DATE'] = df['QUOTE_DATE'].dt.date

	df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'],format='%Y-%m-%d')
	df['EXPIRE_DATE'] = df['EXPIRE_DATE'].dt.date
	return df



read_dir = r'C:\Users\shivampundir\Downloads\spx EOD\VIX EOD\raw_files\TxT Files'
save_dir = r'C:\Users\shivampundir\Downloads\spx EOD\VIX EOD\raw_files\pyarrow'

files_name = os.listdir(read_dir)
for file_name in files_name:
	df = read_custom_csv(os.path.join(read_dir,file_name))
	save_path  = os.path.join(save_dir,'{}.parquet.gzip'.format(file_name.replace('.txt','')))
	df.to_parquet(save_path,compression='gzip')
	print('saved {}'.format(save_path))