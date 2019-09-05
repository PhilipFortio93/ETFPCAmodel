import boto3
from etf_list import *
from datetime import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import simplejson as json
import pickle
import yaml
dev = boto3.session.Session(profile_name='default')

s3 = dev.client('s3') 

my_bucket = 'etf-data-dumps'

def load(key):
    result = s3.get_object(Bucket=my_bucket, Key=key) 
    text = result["Body"].read().decode()
    return text

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def format_data(obj,name):
	obj_df = pd.read_json(obj)
	obj_df = obj_df[['Change %','Date']]
	obj_df.columns = [name,'Date']
	# obj_df[name] = obj_df[name].astype(str)
	
	obj_df['Date'] = obj_df['Date'].astype(str)
	obj_df = obj_df.set_index('Date')
	obj_df.iloc[:,0:1] = obj_df.iloc[:,0:1].applymap(lambda x: float(x[:-1].replace(',','')))
	return obj_df

def format_static_data(obj,name):
	obj_df = pd.DataFrame(obj)
	print(obj_df.head())
	obj_df = obj_df[['Change %','Date']]
	obj_df.columns = [name,'Date']
	# obj_df[name] = obj_df[name].astype(str)
	
	obj_df['Date'] = obj_df['Date'].astype(str)
	obj_df = obj_df.set_index('Date')
	obj_df.iloc[:,0:1] = obj_df.iloc[:,0:1].applymap(lambda x: float(x[:-1].replace(',','')))
	return obj_df

def getISFComp():
	today = datetime.today().strftime('%Y-%m-%d')
	key = '{}/{}/Holdings.json'.format(today,'ISF')
	comp = pd.read_json(load(key))
	comp = comp[['isin','underlyingname','weight']]
	comp = comp.iloc[1:]
	comp.iloc[:,2:3] = comp.iloc[:,2:3].applymap(lambda x: float(x)/100)
	
	return comp

def data_loader():
    print("started the function")
    regions = [
        '/indices/eu-stoxx50-components',
        # '/equities/united-kingdom',
        # '/indices/investing.com-us-500-components',
        ]
    TotalList = []
    for regionurl in regions:
        with open('stock-list'+regionurl.replace("/","-").replace(".","-")+'.json') as outputFile:
            TotalList = TotalList + json.load(outputFile)


    # print('List: ', TotalList)


    return TotalList

def static_load(data):
	today = '2019-08-19'
	datatype = 'PriceHistory'
	path = '/Users/philipfortio/Blockchain ETFs/AWS_Scraper/investing-scraper/data_store/'
	identifier = data['shortname']+'-'+data['ISIN']
	key = '{}/stocks/{}/{}.json'.format(today,identifier,datatype)
	print('key: ', key)

	# raw_data= yaml.safe_load(load(key))
	key = key.replace('/','-')
	file_path = path+key

	print(file_path)
	with open(file_path) as outputFile:
		loaded_data = json.load(outputFile)

	return loaded_data


# def get_cumulative(array):
# 	for 
if __name__ == "__main__":

	file_name = 'data_arrays/raw_stock_data_sector'
	try:
		format_data = pd.read_pickle(file_name+".pkl")
		print("loaded stored pickle")
	except:
		print("loading data")
	try:
		benchmark_df = pd.read_pickle('CUKX.pkl')
	except:
		benchmark = load('2019-08-19/CUKX/Historical.json')
		benchmark_df = pd.DataFrame(yaml.safe_load(benchmark))
		benchmark_df.to_pickle('CUKX.pkl')


	
	benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'].astype(str), format='%d/%b/%Y')
	benchmark_df['Date'] = 	benchmark_df['Date'].dt.strftime("%Y-%m-%d")
	benchmark_df = benchmark_df.set_index('Date')
	fundreturn_df = pd.DataFrame(benchmark_df['NAV'].astype('float'))
	print(type(fundreturn_df))
	fundreturn_df = fundreturn_df.rename(columns={'NAV': 'Benchmark'})
	print(fundreturn_df.head())
	# fundreturn_df = fundreturn_df.rename(columns = 'Benchmark')
	pct_change_bench = pd.DataFrame((fundreturn_df[::-1].pct_change()*100)[::-1])
	
	print(pct_change_bench.head())
	format_data.index= pd.to_datetime(format_data.index.astype(str), format='%b %d, %Y').strftime("%Y-%m-%d")
	print(format_data.head())
	all_data = pct_change_bench.join(format_data)
	print(all_data.head())
	# print(fundreturn_df.head())
	# pct_change_bench = (fundreturn_df[::-1].pct_change()*100)[::-1]
	# pct_change_bench.rename(columns = ['Benchmark'])
	# benchmark_np = (pct_change_bench[:120].values)
	# benchmark_np = benchmark_np[np.newaxis].T
	# print(benchmark_np)
	# print(benchmark_np.shape)



	
	format_data = all_data.dropna()#drop any rows with na
	format_data = format_data[:120]
	# format_data = format_data[['NAV','HSBA-Financial', 'BARC-Financial']]
	# # print('raw_data: ', format_data)
	print(format_data)
	# last_date = str(format_data[-1:].index.values[0])
	# print(last_date)

	data_points = format_data.shape[0]
	stocks = format_data.shape[1]
	names = format_data.columns
	data_array = (format_data.values) # convert to numpy array
	# # mean_adj_data = data_array - np.mean(data_array, axis =0)
	print(data_array)
	# print(data_array.shape)

	# mean_adj_data= all_data
	cov_mat = np.cov(data_array.T)
	variance = data_array.var(0)

	print('cov_mat:' ,cov_mat)
	print(cov_mat.shape)
	print('variance:' ,variance)
	marketvar = variance[0]
	print(variance.shape)

	corr = cov_mat.T/marketvar
	betaplot = corr[0]
	print(corr[0])

	

	

	fig2, ax2 = plt.subplots(1,1)
	fig2.tight_layout()
	
	inds = betaplot.argsort()
	names_sort = names[inds]
	comp1_sort = np.sort(betaplot)
	print(type(names_sort))
	print(names_sort)
	pos = names_sort.get_loc('Benchmark')
	ax2.bar(names_sort,comp1_sort)
	ax2.patches[pos].set_facecolor('#aa3333')
	ax2.set_title('Beta')
	ax2.set_xticklabels(names_sort,rotation=90)
	fig2.subplots_adjust(bottom=0.35)

	for label in ax2.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	plt.show()


