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

def read_parquet(csv_path):
	return pd.read_parquet(csv_path)

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


def built_query_select(conditions,option_chain,threshold):
	"""
	conditions :: {'P_C': 'C', 'B_S': 'B', 'Delta': 0.3, 'Strike': None, 'DTE': 45}
	threshold  :: {'Delta': 0.1, 'Strike': 20, 'DTE': 5,Traded:'Y/N'}
	"""

	p_c = conditions['P_C'].upper()[0]
	if conditions['Delta'] is not None:
		delta_low = conditions['Delta'] - threshold['Delta']
		delta_high = conditions['Delta'] + threshold['Delta']
		query = "{a}_DELTA >= {dl} and {a}_DELTA <= {dh}".format(a=p_c,dl=delta_low,dh=delta_high)
	elif conditions['Strike'] is not None:
		strike_low  = conditions['Strike'] - threshold['Strike']
		strike_high = conditions['Strike'] + threshold['Strike']
		query = "STRIKE >= {dl} and STRIKE <= {dh}".format(dl=strike_low, dh=strike_high)
	dte_low  = conditions['DTE'] - threshold['DTE']
	dte_high = conditions['DTE'] + threshold['DTE']
	query =  query + " and DTE >= {} and DTE <= {}".format(dte_low,dte_high)
	if threshold['Traded'] == 'Y':
		query = query + ' and {a}_VOLUME > 0' .format(a=p_c)
	coarse_selection = option_chain.fillna(0).query(query)

	### Might need to work on this later, keeping it very simple now
	""" Finer Selection, Closet to the DTE """
	if len(coarse_selection) == 0 :
		return
	fine_selection  = coarse_selection.loc[(coarse_selection['DTE']-conditions['DTE']).abs()==(coarse_selection['DTE']-conditions['DTE']).abs().min()]
	if conditions['Delta'] is not None:
		fine_selection = fine_selection.loc[(fine_selection['{a}_DELTA'.format(a=p_c)] - conditions['Delta']).abs() == (fine_selection['{a}_DELTA'.format(a=p_c)] - conditions['Delta']).abs().min()].reset_index(drop=True)
	elif conditions['Strike'] is not None:
		fine_selection = fine_selection.loc[(fine_selection['STRIKE'] - conditions['Strike']).abs() == (fine_selection['STRIKE'.format(a=p_c)] - conditions['Strike']).abs().min()].reset_index(drop=True)
	return fine_selection


def select_bucket(target_list,option_chain):
	"""
	target_list = [ [conditions,threshold],[conditions,threshold]]
	"""
	selections = []
	order_book = {}
	for con,thres in target_list:
		sec = built_query_select(con, option_chain, thres)
		if sec is not None:
			selections.append(sec)
		if sec is None:
			return
		option = get_named_tuple(sec)
		order_book[option.ticker] = {
			'B_S':1 if con['B_S']=='B' else -1
			,'P_C':con['P_C'],'info':option}
	if len(order_book)<len(target_list):
		return
	return  order_book

def buy_option(conditions,option_chain,threshold=None):
	""""select a option based on a set of conditions from an option chain"""
	selection = built_query_select(conditions,option_chain,threshold)
	return selection

def get_named_tuple(option_selection):
	return list(option_selection.itertuples(index=False,name='OptionQuote'))[0]

def select_same_option(ticker,option_chain):
	row = option_chain.loc[option_chain['ticker']==ticker]
	if len(row)!=0:
		return get_named_tuple(row)
	else:
		return

def update_holdings(holdings,option_chain):
	expired_tickers = []
	for ticker in holdings.keys():
		if holdings[ticker]['info'].EXPIRE_DATE < option_chain['QUOTE_DATE'].unique():
			expired_tickers.append(ticker)
			continue
		new_data = select_same_option(ticker, option_chain)
		if new_data is not None:
			holdings[ticker]['info'] = new_data
	for exp in expired_tickers:
		del holdings[exp]
	return holdings

def main_daily_theta_decay():
	""" either the results doesn't make any sense or I am not able to understand the shit """
	path = r"C:\Users\shivampundir\Downloads\spx EOD\Unzipped\*.txt"
	save_path = r'C:\Users\shivampundir\Downloads\spx EOD\thetagang\outputs\1DayChange'
	delta_df = run_delta_enegine(path)
	delta_df.to_csv(os.path.join(save_path,'oneday_change.csv'))

	delta_df = pd.read_csv(os.path.join(save_path,'oneday_change.csv'),index_col=False)
	df = delta_df
	df['return'] = df['Delta_Price']/df['Price']
	df['underlying_return'] = df['Delta_Underlying']/df['Underlying']
	df['Delta'] = (df['Delta']*100).astype(int)
	data = df.groupby(['Put_Call','DTE','Delta']).apply(lambda x:np.sum(x['Volume']*x['return'])/np.sum(x['Volume']))
	data.to_frame(name='DailyReturn').reset_index()
	dtes_to_consider = list(np.arange(2,11,1))
	put_result = data.to_frame(name='DailyReturn').reset_index().query("DTE in @dtes_to_consider and Put_Call in ('P')").pivot_table(index='DTE',
																									 columns='Delta',
																									 values='DailyReturn')
	call_result = data.to_frame(name='DailyReturn').reset_index().query("DTE in @dtes_to_consider and Put_Call in ('C')").pivot_table(index='DTE',
																									 columns='Delta',
																									 values='DailyReturn')

	put_result.reset_index().to_csv(os.path.join(save_path,'put_result.csv'))
	call_result.reset_index().to_csv(os.path.join(save_path,'call_result.csv'))


def volume_analysis():
		daily_volume_concentration = pd.DataFrame([])
		dfs_list = []

		for file_path in files_path[:1]:
			df = read_custom_csv(file_path)

		df = (df
			  .pipe(create_tickers)
			  .pipe(add_mid_price)
			  )

		print(df.head().to_string())

		old_chain = get_option_chain(df.loc[df.QUOTE_DATE == pd.to_datetime('2018-01-02')])
		new_chain = get_option_chain(df.loc[df.QUOTE_DATE == pd.to_datetime('2018-01-03')])



def long_short_holdings(holdings):
	long_delta , long_theta ,long_price  , short_delta , short_theta , short_price = 0,0,0,0,0,0
	for ticker,pos in holdings.items():
		if pos['P_C'] == 'P' and pos['B_S'] == 1:
			short_delta += pos['info'].P_DELTA
			long_price  += pos['info'].P_MID
			long_theta  += abs(pos['info'].P_THETA)*-1
		if pos['P_C'] == 'P' and pos['B_S'] == -1:
			long_delta += pos['info'].P_DELTA*-1
			short_price  += pos['info'].P_MID
			short_theta += abs(pos['info'].P_THETA)
		if pos['P_C'] == 'C' and pos['B_S'] == 1:
			long_delta += pos['info'].C_DELTA
			long_price  += pos['info'].C_MID
			long_theta  += abs(pos['info'].C_THETA)*-1
		if pos['P_C'] == 'C' and pos['B_S'] == -1:
			short_delta += pos['info'].C_DELTA*-1
			short_price  += pos['info'].C_MID
			short_theta  += abs(pos['info'].C_THETA)
	return long_delta , long_theta ,long_price  , short_delta , short_theta , short_price

def run_portfolio_analytics(portfolio):
	portfolio_numbers = []
	for date,holdings in portfolio.items():
		long_delta , long_theta ,long_price  , short_delta , short_theta , short_price = long_short_holdings(holdings)
		portfolio_numbers.append([date,long_delta , long_theta ,long_price  , short_delta , short_theta , short_price])

	result = pd.DataFrame(portfolio_numbers,columns=['QUOTE_DATE','long_delta' , 'long_theta' ,'long_price'  , 'short_delta' , 'short_theta' , 'short_price'])
	return result


def get_next_exposures(df):
	df['Value'] = df['long_price'] + df['short_price']
	df['Delta'] = df['long_delta'] + df['short_delta']
	df['Theta'] = df['long_theta'] + df['short_theta']
	return df


def work_transaction(cash,trade_order):
	"""returns the transaction info
		Boolean , Str , float
		- True / False ( If True transaction possible, False transaction is not possible)
		- ND/NC ( Net Debit / Net Credit)
		- Final Cash in Account
	"""
	order_book = deepcopy(trade_order)
	curr_cash = 0
	credit_type = None
	possible_flag = False
	for ticker,pos in order_book.items():
		if pos['P_C'] == 'P' and pos['B_S'] == 1:
			curr_cash -= pos['info'].P_MID
		if pos['P_C'] == 'P' and pos['B_S'] == -1:
			curr_cash += pos['info'].P_MID
		if pos['P_C'] == 'C' and pos['B_S'] == 1:
			curr_cash -= pos['info'].C_MID
		if pos['P_C'] == 'C' and pos['B_S'] == -1:
			curr_cash += pos['info'].C_MID
	if   curr_cash > 0:
		### TODO , Check for BRP
		credit_type = 'NC'
		possible_flag = True
#		cash = cash + curr_cash
	elif curr_cash < 0:
		if cash + curr_cash >= 0:
			credit_type = 'ND'
			possible_flag = True
#			cash = cash + curr_cash
		else:
			possible_flag = False
	return possible_flag,credit_type,curr_cash

def liquidate_order_book(order_book):
	for ticker,pos in order_book.items():
		order_book[ticker]['B_S'] = pos['B_S']*-1
	return order_book


def get_underlying(option_chain):
	return option_chain.set_index('QUOTE_DATE')['UNDERLYING_LAST'].drop_duplicates().to_dict()

def is_liquidation_meet(liquidate_conditions,curr_holdings,trade_log,date):
	"""checks out if the liquidation condition has been meet or now

		- Option has Expired
	"""

	## ToDO
	## Calender Spread
	## Last Trading Day

	if len(curr_holdings)==0:
		return False
	dte_remaining = min(map(lambda x: x['info'].DTE, curr_holdings.values()))
	if liquidate_conditions['DTE'] >= dte_remaining:
		return True
	return False
	#_, credit_type, liquidation_amount = work_transaction(1_000_000, liquidate_order_book(deepcopy(curr_holdings)))
#	return  dte_remaining,credit_type,liquidation_amount

if __name__ == "__main__":

	portfolio = {}
	underlying = {}
	curr_holdings = {}
	cash = 10_000
	liquidation_value = {}
	order_number = 1
	trade_log = {}
	target_list = [
		  [ {'P_C'   : 'C'  , 'B_S'   : 'B'    , 'Delta': 0.5   , 'Strike': None , 'DTE': 20},
		    {'Delta' : .1   , 'Strike': 20     , 'DTE'  : 2     , 'Traded': 'Y'}]
		# , [ {'P_C'   : 'C'  , 'B_S    ': 'B'   , 'Delta': 0.5   , 'Strike': None , 'DTE': 10},
		#     {'Delta' : .1   , 'Strike': 20     , 'DTE'  : 2     , 'Traded': 'Y'}]
		# , [{'P_C': 'P', 'B_S': 'S', 'Delta': -0.5, 'Strike': None, 'DTE': 45},
		#     {'Delta': .1, 'Strike': 20, 'DTE': 5, 'Traded': 'Y'}]
		# , [{'P_C': 'P', 'B_S': 'B', 'Delta': -0.4, 'Strike': None, 'DTE': 45},
		#     {'Delta': .1, 'Strike': 20, 'DTE': 5, 'Traded': 'Y'}]
	       ]

	liquidate_conditions = {
		  'DTE'    : 2
		, 'Loss'   : 0.20
		, 'Profit' : 0.25 }

	is_trade_conditions_meet = True

	path = r"C:\Users\shivampundir\Downloads\spx EOD\VIX EOD\raw_files\pyarrow\*.gzip"
	files_path = glob(path)
	for file_path in files_path[:]:
		df = read_parquet(file_path)
		df = (
			  df
			  .pipe(create_tickers)
			  .pipe(add_mid_price)
			  )

		grouped = df.groupby(['QUOTE_DATE'])
		for grp in grouped:
			date, raw_data = grp[0], grp[1]
			option_chain = get_option_chain(raw_data)
			print('working for date {}'.format(date.strftime("%d/%m/%Y")))
			underlying.update(get_underlying(option_chain))
			if len(curr_holdings)==0 and is_trade_conditions_meet:
				# generate order
				# logic what needs to be bought should be written here
#				print('No Position , Buying')
				trade_targets = select_bucket(target_list , option_chain)
				if trade_targets is not None:
					possible_flag , credit_type , amount = work_transaction(cash,trade_targets)
					if possible_flag:
						cash 		           += amount 		   # cash
						curr_holdings           = trade_targets  # current holdings
						trade_log[order_number] =  {'date'         : date
													,'credit_type' : credit_type
													,'order'	   : deepcopy(curr_holdings)
													,'amount'	   : amount}
						order_number           += 1			   # order number
						liquidation_value[date] = cash - amount ## We can keep it equal to starting cash also,

			if len(curr_holdings)>0:
				# if sum(map(lambda x: x['info'] is None, curr_holdings.values()))>0:
				# 	print('Skipped for Date {}'.format(date.strftime("%Y-%m-%d")))
				# 	continue
				curr_holdings = update_holdings(curr_holdings, option_chain)
				liquidation_value[date] = cash + work_transaction(1_000_000,liquidate_order_book(deepcopy(curr_holdings)))[2]

				if is_liquidation_meet(liquidate_conditions,deepcopy(curr_holdings),trade_log,date):
					liquidation_order = liquidate_order_book(deepcopy(curr_holdings))
					possible_flag, credit_type, amount = work_transaction(1_000_000,deepcopy(liquidation_order))
					curr_holdings 			 = {}
					cash 		 			+= amount # cash
					trade_log[order_number]  = {  'date'        : date
												, 'credit_type'	: credit_type
												, 'order' 		: deepcopy(liquidation_order)
												, 'amount'		: amount
												}
					liquidation_value[date]  = cash ###
					order_number 			+= 1			   # order number

			portfolio[date] = deepcopy(curr_holdings)
	res = run_portfolio_analytics(deepcopy(portfolio))
	res = res.pipe(get_next_exposures)
	underlying = pd.Series(underlying).to_frame(name='Underlying')
	# res.set_index('QUOTE_DATE')[['Theta','Delta','Value']].plot(subplots=True,marker='o')
	# plt.xlabel('')
	# plt.show()

	res  = res.set_index('QUOTE_DATE')
	res['Underlying'] =  underlying
	lv = pd.Series(liquidation_value).to_frame(name='Liquidation Value')
	res['Liquidation_Value'] = lv
	res.index = pd.to_datetime(res.index,format='%Y-%m-%d')

	plotdf = deepcopy(res[['Liquidation_Value', 'Underlying']])
	plotdf['Liquidation_Value'] = plotdf['Liquidation_Value']/plotdf.loc[plotdf.index==plotdf['Liquidation_Value'].first_valid_index(),'Liquidation_Value'].values[0]
	plotdf['Underlying'] = plotdf['Underlying'] / \
							  plotdf.loc[plotdf.index == plotdf['Underlying'].first_valid_index(), 'Underlying'].values[0]

	# plotdf.plot() ; plt.show()
	# monthly_return = plotdf.fillna(method='ffill').pct_change().groupby(pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1)

	yearly_return  = plotdf.fillna(method='ffill').pct_change().groupby(pd.Grouper(freq='Y')).apply(lambda x: (1 + x).prod() - 1)

## This code mess things up.
	# dates = list(portfolio.keys())
	# for dt in dates:
	# 	pt = portfolio[dt]
	# 	pt = liquidate_order_book(pt)
	# 	credit_event = work_transaction(100_000, pt)
	# 	print(dt, credit_event)

"""
	for p_c in ['P','C']:
		for strike in [2750,2800,3000]:
			for DTE in [5,10,21,45]:
				conditions = {'P_C': p_c, 'B_S': 'B', 'Delta':None, 'Strike': strike, 'DTE': DTE}
				threshold = {'Delta': 0.05, 'Strike': 20, 'DTE': 5,'Traded':'Y'}
				out = built_query_select(conditions, new_chain, threshold)
				print('P/C : ',p_c," ,DTE : ",DTE," ,Strike : ",strike)
				if out is not None:
					print(out.to_string())
				else:
					print('No Trade')
"""


def run_trade_log_analytics(trade_log):
	
	trade_list_info = []
	for t_number,trade_info in trade_log.items():
		trade_list_info.append((trade_info['date'],trade_info['credit_type'],trade_info['amount']))
	return pd.DataFrame(trade_list_info,columns=['QUOTE_DATE','Credit_Type','Amount'])
		
