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

	file_name = 'data_arrays_sx5e/raw_stock_data_sector_sx5e'
	try:
		format_data = pd.read_pickle(file_name+".pkl")
		print("loaded stored pickle")
	except:
		print("loading data")
		# weights = np.full((1,10),0.1)
		# stocks = 5
		names = []
		today = datetime.today().strftime('%Y-%m-%d')
		# ftsecomp = getISFComp()
		# weights = ftsecomp[['weight']][:stocks].values
		# weight_adj = sum(weights)
		# weights = weights/weight_adj
		# print(weights[1][0])
		
		data_list = data_loader()

		for each in data_list[0:1]:
			# print(each)
			identifier = each['shortname']+'-'+each['sector']
			names.append(identifier)
			# key = '{}/{}/PriceHistory.json'.format(today,each['name'])
			format_data = format_static_data(static_load(each),identifier)#format_data(load(key),each['name'])
			# print(format_data)
			format_data = format_data.iloc[::-1]
		i =0
		for each in data_list[1:]:
			names.append(each['shortname'])
			# key = '{}/{}/PriceHistory.json'.format(today,each['shortname'])
			# print(key)
			obj = static_load(each)
			print(each['shortname'])
			print(each['sector'])
			# format_data_next = format_static_data(obj, each['shortname'])
			obj_df = pd.DataFrame(obj)
			# format_data_next = format_static_data(static_load(each),each['shortname'])
			# print(format_data_next.head())
			identifier = each['shortname']+'-'+each['sector']
			# identifier = each['sector']
			obj_df = obj_df[['Change %','Date']]
			obj_df['Date'] = obj_df['Date'].astype(str)
			obj_df.columns = [identifier,'Date']
			format_data_next = obj_df.set_index('Date')
			print(format_data_next.iloc[:,0:1])

			format_data_next.iloc[:,0:1] = format_data_next.iloc[:,0:1].applymap(lambda x: float(x[:-1].replace(',','')))#*weights[i][0]) #remove commas
			format_data_next = format_data_next.loc[abs(format_data_next[identifier]) < 90] # filter out erroneous data for now
			format_data_next = format_data_next.iloc[::-1]
			# print(format_data_next.head())
			# print(format_data_next)
			format_data = format_data.join(format_data_next,on='Date', how='left')	
			i = i + 1
			# print (format_data.head())
		format_data.to_pickle(file_name+".pkl")
		
	


	format_data = format_data.dropna()#drop any rows with na
	format_data = format_data[-120:]
	print('raw_data: ', format_data.head())
	data_points = format_data.shape[0]
	stocks = format_data.shape[1]

	names = format_data.columns
	



	format_data = format_data/stocks


	

	# ISF = format_data[['ISF']]
	# print(ISF.head())
	# # ISF = ISF.iloc[::-1]
	# ISF['cum'] = ISF['ISF'].cumsum()
	# market = ISF[['cum']].values
	

	# ax[0].plot(market)
	# ax[0].set_title("FTSE 100 return")
	# format_data = format_data.drop('ISF', 1) #Remove market column

	# # print('data \n', format_data.head())

	data_array = (format_data.values) # convert to numpy array

	dailyreturns = data_array.sum(axis=1)
	stockperf = format_data.sum().sort_values()
	# # dailyreturns = data_array.dot(weights) # portfolio weighting
	# # print('data set \n', data_array)
	# # print(dailyreturns)
	# print(dailyreturns.shape)
	cumulativereturns = np.cumsum(dailyreturns)

	
	print(data_array.shape)
	# # plt.show()

	mean_adj_data = data_array - np.mean(data_array, axis =0)
	# print('mean adj data: \n',mean_adj_data)

	cov_mat = np.cov(mean_adj_data.T) #covariance also an option
	# cov_mat = np.corrcoef(mean_adj_data.T) #covariance also an option
	# print('Covariance matrix \n%s' %cov_mat)

	eig_vals, eig_vecs = np.linalg.eig(cov_mat)



	print('Eigenvectors \n%s' %eig_vecs)
	print('\nEigenvalues \n%s' %eig_vals)
	eigen_sum = eig_vals.sum()
	print('eigen_sum = ',eigen_sum)
	multi_eig = np.column_stack((eig_vecs.T[0],eig_vecs.T[1],eig_vecs.T[2],eig_vecs.T[3]))
	# # print('\multi_eig \n%s' %multi_eig)

	print('stockperf: ',stockperf)

	i = 0
	while i < 4:
		fig2, ax2 = plt.subplots(1,1)
		fig2.tight_layout()

		inds = eig_vecs.T[i].argsort()
		names_sort = names[inds]
		comp1_sort = np.sort(eig_vecs.T[i])
		ax2.bar(names_sort,comp1_sort)
		ax2.set_title('Component '+str(i))
		ax2.set_xticklabels(names_sort,rotation=90)
		i = i + 1


	fig, ax = plt.subplots(5,1)

	ax[1].plot(cumulativereturns)
	ax[1].set_title('Portfolio returns')
	
	# # plotting 
	# # fig, ax = plt.subplots(2,5)
	# # num_bins = 100
	# # i = 0
	# # while (i < 2):
	# # 	j = 0
	# # 	while( j < 5):
	# # 		print(data_array[:, i*5 + j])
	# # 		n, bins, patches = ax[i,j].hist(data_array[:, i*5 + j], num_bins, density=1)
	# # 		ax[i,j].set_title(names[i*5+j])
	# # 		j = j + 1
	# # 	i = i +1

	# # plt.show()

	# This applies a linear transform on the data into a basis in which the correlation matrix is diagonal
	principle_two = mean_adj_data.dot(multi_eig) 
	print('linear transform \n%s' %principle_two)
	# This reconstructs an estimate of the returns based on the decomposed eigenvectors
	recon1 = np.outer(principle_two.T[0],multi_eig.T[0]) #+ np.mean(data_array, axis =0)
	print('recon1 \n%s' %recon1)

	recon2 = np.outer(principle_two.T[1],multi_eig.T[1]) #+ np.mean(data_array, axis =0)
	# print('recon2 \n%s' %recon2)

	recon = recon1 + recon2 + np.mean(data_array, axis =0)
	print('total \n%s' %recon)
	print('initial dat \n%s'%data_array)
	print(recon1.shape)
	cumrecon = np.cumsum(recon)

	print('Actual returns',dailyreturns)
	print('reconstructed', recon.sum(axis=1))
	reconerror = abs(recon.sum(axis=1) - dailyreturns)
	print('recon error for 2 manual componenets: ',np.sum(reconerror)/data_points)
	ax[0].plot(cumrecon)
	ax[0].set_title("manual PCA recon")

	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data_array)
	# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])

	reconstruction = pca.inverse_transform(principalComponents) 
	reconstructionDf = pd.DataFrame(data = reconstruction, columns = names)

	# print(principalDf)
	# print(pca.components_[0])
	# print(pca.explained_variance_)
	# print(format_data)
	# print(reconstructionDf)

	recondailyreturns = np.cumsum(reconstructionDf.values.sum(axis=1))
	print('pca shape: ',reconstructionDf.shape)
	reconerror = abs(reconstructionDf.values.sum(axis=1)- dailyreturns)
	# print(reconerror)
	print('recon error for 2 componenets: ',np.sum(reconerror)/data_points)

	ax[2].plot(recondailyreturns)
	ax[2].set_title("2 component recon")

	pca = PCA(n_components=10)
	principalComponents = pca.fit_transform(data_array)
	reconstruction = pca.inverse_transform(principalComponents) 
	reconstructionDf = pd.DataFrame(data = reconstruction
             , columns = names)
	recondailyreturns = np.cumsum(reconstructionDf.values.sum(axis=1))
	ax[3].plot(recondailyreturns)
	ax[3].set_title("10 component recon")

	recondailyreturns = np.cumsum(reconstructionDf.values.sum(axis=1))

	reconerror = abs(reconstructionDf.values.sum(axis=1)- dailyreturns)
	# print(reconerror)
	print('recon error for 10 componenets: ',np.sum(reconerror)/data_points)

	print('explained var 10 componenets: ',pca.explained_variance_.sum()/eigen_sum)

	pca = PCA(n_components=14)
	principalComponents = pca.fit_transform(data_array)
	# print(principalComponents)
	reconstruction = pca.inverse_transform(principalComponents) 
	reconstructionDf = pd.DataFrame(data = reconstruction
             , columns = names)
	# print(reconstructionDf)
	recondailyreturns = np.cumsum(reconstructionDf.values.sum(axis=1))
	ax[4].plot(recondailyreturns)
	ax[4].set_title("14 component recon")

	recondailyreturns = np.cumsum(reconstructionDf.values.sum(axis=1))

	reconerror = abs(reconstructionDf.values.sum(axis=1)- dailyreturns)
	# print(reconerror)
	print('recon error for 14 componenets: ',np.sum(reconerror)/data_points)

	print('explained var for 14 comp: ',pca.explained_variance_.sum()/eigen_sum)
	eig_vecsDf = pd.DataFrame(data = eig_vecs)
	# print('Eigenvectors \n%s' %eig_vecsDf)


	plt.show()


