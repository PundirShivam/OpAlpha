import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os


def read_custom_csv(csv_path):
	#R"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\spx_eod_201801.txt"
	df = pd.read_csv(csv_path,delimiter=", ")
	df.columns = df.columns.str.replace("[","").str.replace("]","")

	df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'],format='%Y-%m-%d')
	df['QUOTE_DATE'] = df['QUOTE_DATE'].dt.date

	df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'],format='%Y-%m-%d')
	df['EXPIRE_DATE'] = df['EXPIRE_DATE'].dt.date
	return df


#df.groupby(['QUOTE_DATE','EXPIRE_DATE'])[['P_VOLUME','C_VOLUME']].sum().plot()

#plt.show()


def _total_volume(tdf):
	tdf['total_volume'] = tdf['C_VOLUME'] + tdf['P_VOLUME']
	return tdf

def _dte_custom(tdf):
	x = tdf['DTE']
	conditions =  [ x == 0 , (x > 0) &( x < 6), x > 5 ]
	select = [ "0DTE" , "1-5DTE" , "Above5DTE"]
	tdf['dte_class'] = np.select( conditions , select)
	return tdf	


def _daily_volume_fraction(tdf):
	xx = tdf.groupby(['QUOTE_DATE'])['total_volume'].sum()
	yy = tdf.set_index('QUOTE_DATE')['total_volume'].div(xx)
	yy = yy.reset_index()
	tdf['fraction_volume'] = yy['total_volume']
	return tdf  

## Good Looking Chart but difficult to read

"""
(
df.groupby(['QUOTE_DATE','EXPIRE_DATE'])[['P_VOLUME','C_VOLUME']].sum()
.reset_index()
.pipe(_total_volume)
.pipe(_daily_volume_fraction)
.pivot_table(columns='QUOTE_DATE',index='EXPIRE_DATE',values='fraction_volume')
.plot(kind='bar',stacked=True))
plt.show()
"""
## Trying on DTEs


def volume_fraction(df):
	return ( df.groupby(['QUOTE_DATE','DTE'])[['P_VOLUME','C_VOLUME']].sum()
  		.reset_index()
		.pipe(_total_volume)
		.pipe(_dte_custom)
		.pipe(_daily_volume_fraction)
		.groupby(['QUOTE_DATE','dte_class'])['fraction_volume'].sum()
		.reset_index()
		.pivot_table(index='QUOTE_DATE',columns='dte_class',values='fraction_volume'))


def create_tickers(df):
	"""adds tickers for calls and puts"""
	df['ticker'] = pd.to_datetime(df['EXPIRE_DATE']).dt.strftime("%d%m%Y")+df['STRIKE'].astype(int).astype(str)
	return df


def add_mid_price(df):
	"""adds mid price to call and put """
	df['C_MID'] = (df['C_BID']+df['C_ASK'])/2
	df['P_MID'] = (df['P_BID']+df['P_ASK'])/2
	return df

def add_underlying_delta(df):
	""" adds percent change and dollar change of the underlying from the pervious day"""
	underlying_prc = df.set_index('QUOTE_DATE')['UNDERLYING_LAST'].drop_duplicates()
	delta_df = pd.DataFrame([])
	delta_df['underlying_delta'] = underlying_prc.diff()
	delta_df['underlying_pct_change'] = underlying_prc.pct_change()
	df['underlying_dlr_change'] = df['QUOTE_DATE'].map(delta_df['underlying_delta'])
	df['underlying_pct_change'] = df['QUOTE_DATE'].map(delta_df['underlying_pct_change'])
	return df

def calc_change_in_option_chain(old_option_chain,new_option_chain):
	pass


def get_option_chain(df):
	"""returns the data in option chain format , better udfse it with generatror """
	cols = ['QUOTE_DATE','ticker','UNDERLYING_LAST','EXPIRE_DATE','DTE','C_MID','C_IV','C_VOLUME','C_DELTA','C_THETA','C_VEGA','C_GAMMA','STRIKE','P_MID','P_IV','P_VOLUME','P_DELTA','P_THETA','P_VEGA','P_GAMMA']
	return df.filter(cols).sort_values(['EXPIRE_DATE', 'STRIKE'])


def describe_option_chain(df):
	pass

def get_change_in_option_chain(old_chain,new_chain):
	common_tickers = set(old_chain['ticker'].drop_duplicates().tolist()).intersection(set(new_chain['ticker'].drop_duplicates().tolist()))
	oc = old_chain.query('ticker in @common_tickers')
	nc = new_chain.query('ticker in @common_tickers')
	delta = (nc.set_index('ticker').fillna(0)-oc.set_index('ticker').fillna(0))
	delta = (delta
			 .drop(['QUOTE_DATE','EXPIRE_DATE','STRIKE'],axis=1)
			 .add_prefix('DIFF_')
			 .reset_index()
			 )
	return delta

def add_options_vol_and_price_delta(old_chain,new_chain):
	"""adds daily volume and price delta for each option in option chain"""
	delta = get_change_in_option_chain(old_chain,new_chain)
	delta_forward_looking = old_chain.merge(delta,on='ticker')
	cols = ['QUOTE_DATE','DTE','Volume','Delta_Volume','Underlying','Delta_Underlying','Price','Delta_Price','Delta','Delta_Delta','Put_Call']
	calls = delta_forward_looking[['QUOTE_DATE','DTE','C_VOLUME','DIFF_C_VOLUME','UNDERLYING_LAST','DIFF_UNDERLYING_LAST','C_MID','DIFF_C_MID','C_DELTA','DIFF_C_DELTA']]
	puts  = delta_forward_looking[['QUOTE_DATE','DTE','P_VOLUME','DIFF_P_VOLUME','UNDERLYING_LAST','DIFF_UNDERLYING_LAST','P_MID','DIFF_P_MID','P_DELTA','DIFF_P_DELTA']]
	calls['Put_Call'] = 'C'
	puts['Put_Call']  = 'P'
	calls.columns = cols
	puts.columns = cols
	change_df = pd.concat([calls,puts],axis=0)
	cols_reorder = ['QUOTE_DATE', 'DTE','Put_Call','Volume', 'Delta_Volume', 'Underlying', 'Delta_Underlying', 'Price', 'Delta_Price',
			'Delta', 'Delta_Delta']
	change_df = change_df[cols_reorder]
	"""Intend is to reduce the non traded price quotes in order to save computation time and effort"""
	change_df = (change_df
		.loc[(~change_df.Volume.isna() & change_df.Volume != 0)]
		)
	return change_df

def run_delta_enegine(path=None):
	##""" path = r"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\*.txt """
	if path is None:
		path = r"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\*.txt"
	files_path = glob(path)
	delta_dfs_list = []
	new_chain = None
	old_chain = None
	for file_path in files_path[:]:
		df = read_custom_csv(file_path)

		df = (
			  df
			  .pipe(create_tickers)
			  .pipe(add_mid_price)
			  )

		grouped = df.groupby(['QUOTE_DATE'])
		for grp in grouped:
			date,raw_data = grp[0],grp[1]
			new_chain = get_option_chain(raw_data)
			if old_chain is not None:
				delta_dfs_list.append(add_options_vol_and_price_delta(old_chain,new_chain))
				print('Ran for date {date}'.format(date=date.strftime('%Y-%m-%d')))
			old_chain = new_chain
	delta_df = pd.concat(delta_dfs_list,axis=0)
	delta_df = delta_df.reset_index(drop=True)
	return delta_df

class option:
	""" class option """
	def __init__(self,QUOTE_DATE,ticker,UNDERLYING_LAST,EXPIRE_DATE,DTE,C_MID,C_IV,C_VOLUME,C_DELTA,C_THETA,C_VEGA,C_GAMMA,STRIKE,P_MID,P_IV,P_VOLUME,P_DELTA,P_THETA,P_VEGA,P_GAMMA):

		self.QUOTE_DATE = QUOTE_DATE
		self.ticker = ticker
		self.UNDERLYING_LAST = UNDERLYING_LAST
		self.EXPIRE_DATE = EXPIRE_DATE
		self.DTE = DTE
		self.C_MID = C_MID
		self.C_IV = C_IV
		self.C_VOLUME = C_VOLUME
		self.C_DELTA = C_DELTA
		self.C_THETA = C_THETA
		self.C_VEGA = C_VEGA
		self.C_GAMMA = C_GAMMA
		self.STRIKE = STRIKE
		self.P_MID = P_MID
		self.P_IV = P_IV
		self.P_VOLUME = P_VOLUME
		self.P_DELTA = P_DELTA
		self.P_THETA = P_THETA
		self.P_VEGA = P_VEGA
		self.P_GAMMA  = P_GAMMA

	def __eq__(self,other):
		return self.ticker == other.ticker

	def __str__(self):
		return self.ticker
#		return "Quote Date : {} - Expire Date : {} - Strike : {} ".format(self.QUOTE_DATE.strftime('%Y-%m-%d'),self.EXPIRE_DATE.strftime('%Y-%m-%d'),self.STRIKE)

	def __repr__(self):
		return self.ticker
#		return "Quote Date : {} - Expire Date : {} - Strike : {} ".format(self.QUOTE_DATE.strftime('%Y-%m-%d'),self.EXPIRE_DATE.strftime('%Y-%m-%d'),self.STRIKE)


class trade:
	
	@classmethod
	def BTO(self,P_C,DELTA,STRIKE,DTE):
		pass
	
	@classmethod
	def STO():
		pass

	@classmethod
	def BTC():
		pass

	@classmethod
	def STC():
		pass


def plot_option(data):
	"""plots dataframe assuming it consists of option data"""
	cols = ['DATETIME','Value','Underlying','IV','Delta','Theta','Gamma','Vega','Rho','Volume']
	plt_data = data.filter(cols)
	plt_data = plt_data.set_index('DATETIME')
	plt_data.plot(kind='line',marker='.',subplots=True,figsize=(12,8))
	plt.show()
	

if __name__ == "__main__":

	path = r"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\*.txt"
	files_path = glob(path)
	for file_path in files_path[:1]:
		df = read_custom_csv(file_path)

	df = (df
		  .pipe(create_tickers)
		  .pipe(add_mid_price)
		  )
	grouped = df.groupby(['QUOTE_DATE'])
	for grp in grouped:
		date, raw_data = grp[0], grp[1]
		new_chain = get_option_chain(raw_data)
		for row in new_chain.itertuples(index=False, name=None):
			option_new = option(*row)
			pass

	data = df.loc[df.ticker=='160220182810']
	data = data[['QUOTE_DATE','C_MID','UNDERLYING_LAST','C_IV','C_DELTA','C_THETA','C_GAMMA','C_VEGA','C_RHO','C_VOLUME']]#
	data.columns = 	['DATETIME', 'Value', 'Underlying', 'IV', 'Delta', 'Theta', 'Gamma', 'Vega', 'Rho', 'Volume']

#			print(option_new)
	

