import boto3
from etf_list import *
from datetime import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import simplejson as json
import pickle
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.decomposition import PCA
from sklearn import mixture
from matplotlib.colors import LogNorm
# import keras
# from keras.models import load_model



class AnimatedScatter(object):
	def __init__(self, label=[],frames=1,standout=[2],data_dir='data_arrays/pca_weights_',save_fig=False):
		# self.stream = self.data_stream()
		self.frames = frames
		self.fig, self.ax = plt.subplots()
		plt.tight_layout()
		self.data_dir = data_dir
		with open('data_arrays/title_list.pkl', 'rb') as f:
			self.title_list = pickle.load(f)

		self.label = label
		self.standout = standout

		self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000, init_func= self.setup_plot, blit=False, frames=xrange(self.frames))
		# self.indexlabel = indexlabel
		if(save_fig==True):
			self.ani.save('line.gif', dpi=300, writer='imagemagick')

	
	def setup_plot(self):
		# x,y = next(self.stream).T
		weights_file = self.data_dir+'0.pkl'
		weights_data = pd.read_pickle(weights_file)
		predicted_array = (weights_data.values)
		x = predicted_array[:,:1]
		y = predicted_array[:,1:]

		gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(predicted_array)
		# display predicted scores by the model as a contour plot
		xmesh = np.linspace(-1., 1.,200)
		ymesh = np.linspace(-1., 1.,200)
		X, Y = np.meshgrid(xmesh, ymesh)
		XX = np.array([X.ravel(), Y.ravel()]).T
		Z = gmm.score_samples(XX)
		print(max(Z))
		Z = Z.reshape(X.shape)*50

		self.CS = plt.contourf(X, Y, Z	,cmap='jet', 
                 levels=np.arange(-1000, 1000, 50))
		self.CB = plt.colorbar(self.CS, shrink=0.8, extend='both')

		self.scat = self.ax.scatter(x,y,marker='x')
		self.ax.axis([-1,1,-1,1])


		self.title = self.ax.text(0.5,1, self.title_list[0], bbox={'facecolor':'w', 'alpha':1, 'pad':5},transform=self.ax.transAxes, ha="center")
		# print(self.label)

		# self.ax.annotate(self.label,(x[self.indexlabel],y[self.indexlabel]),xytext=(x[self.indexlabel],y[self.indexlabel]), fontsize=9)
		# annotation.set_animated(True)
		# self.annotate = self.ax.annotate(self.label[self.standout],(x[self.standout],y[self.standout]), fontsize=15, fontweight=1000)
		self.signy = np.sign(x[2])
		self.signx = np.sign(y[2])
		self.annotations = []
		for i, txt in enumerate(self.label):
		# 	self.annotations[i].xy = (x[i],y[i])
		# 	# self.annotations[i].set_animated(True)
			if i in self.standout:
				self.annotations.append(self.ax.annotate(txt,(x[i],y[i]), fontsize=9, fontweight=1000))
			else:
				self.annotations.append(self.ax.annotate(txt,(x[i],y[i]), fontsize=9))
		# 	print(self.annotations[i].xy)
		# print(self.annotations)


		return self.scat,  self.title, self.annotations, self.CS, self.CB,#self.annotate,

	def data_stream(self,i):
		weights_file = self.data_dir +str(i) +'.pkl'
		weights_data = pd.read_pickle(weights_file)



		predicted_array = (weights_data.values)
		# k = 0
		while i < self.frames:
			yield np.c_[predicted_array[:,:1],predicted_array[:,1:]]
			# k = k + 1

	def update(self,i):
		print('frame: ',i)
		if i == self.frames-1:
			print("resetting")
			self.ax.clear()
			self.CB.remove()
			# self.annotations = []
			# for i, txt in enumerate(self.label):
			# # 	self.annotations[i].xy = (x[i],y[i])
			# # 	# self.annotations[i].set_animated(True)
			# 	self.annotations.append(self.ax.annotate(txt,(x[i],y[i]), fontsize=11))

		data = next(self.data_stream(i))
		if np.sign(data[:,1:][2]) != self.signy:
			data[:,1:] = -1*data[:,1:]
		if np.sign(data[:,:1][2]) != self.signx:
			data[:,:1] = -1*data[:,:1]

		x = data[:,:1]
		y = data[:,1:]
		self.scat.set_offsets(np.c_[data[:,:2]])

		# self.scat = self.ax.scatter(x,y)
		# self.ax.annotate(self.label,(x[self.indexlabel],y[self.indexlabel]),xytext=(x[self.indexlabel],y[self.indexlabel]), fontsize=9)
		# self.annotate.set_position((data[:,:1][self.standout],data[:,1:][self.standout]))
		self.title.set_text(self.title_list[i])
		for inx, txt in enumerate(self.label):
		# 	print(inx)
		# 	print(data[:,:1][inx])
			self.annotations[inx].set_position((data[:,:1][inx],data[:,1:][inx]))

		# for coll in self.CS.collections: 
		#     plt.gca().collections.remove(coll) 
		for c in self.CS.collections:
			c.remove() 

		
		gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(data)
		# display predicted scores by the model as a contour plot
		xmesh = np.linspace(-1., 1.,200)
		ymesh = np.linspace(-1., 1.,200)
		X, Y = np.meshgrid(xmesh, ymesh)
		XX = np.array([X.ravel(), Y.ravel()]).T
		Z = gmm.score_samples(XX)
		Z = Z.reshape(X.shape)*50
		self.CS = plt.contourf(X, Y, Z, cmap='jet',
                  levels=np.arange(-1000, 1000, 50))
		# print(self.annotations[2])
		
		return self.scat, self.title, self.annotations,self.CS, self.CB, #self.annotate,
# Returns a short sequential model
def create_model(inputshape, outputshape):
  model = tf.keras.models.Sequential([
    keras.layers.Dense(outputshape, activation=tf.keras.activations.softmax, input_shape=(inputshape,)),
    # keras.layers.Dropout(0.5),
    # keras.layers.Dense(outputshape, activation=tf.keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])

  return model

if __name__ == "__main__":
	main_dir = 'data_arrays'
	file_name = main_dir+'/raw_stock_data_sector'
	models_file = 'saved_models/model_train'
	title_list =[]
	format_data = pd.read_pickle(file_name+".pkl")
	print("loaded stored pickle")
	all_data = format_data.dropna()#drop any rows with na
	print(all_data.shape)
	columns = list(format_data.columns)
	print(columns)
	one_follow = [x if x[:x.find("-")] == 'BARC' else '' for x in columns]
	# label_index = 3
	total_frames = 65
	k = 0
	epochs = 200
	runmodel = False

	if runmodel == True:
		while k < total_frames:
			print('k =', k)
			format_data = all_data[15*k:15*k+120]
			try:
				last_date = str(format_data[-1:].index.values[0])

			except:
				last_date ='error'

			print('final_date: ', last_date)
			title_list.append(last_date)
			print('raw_data: ', format_data.head())
			data_points = format_data.shape[0]
			stocks = format_data.shape[1]
			
			sectors_list = [x[x.find("-")+1:] for x in columns]
			names = [x[:x.find("-")] for x in columns]
			
			# print(type(sectors_list))

			# print('columns: ', sectors_list)
			data_array = (format_data.values).T # convert to numpy array
			print(data_array)
			print(data_array.shape)

			split_pcnt = 75
			test_split =int(data_array.shape[0]*split_pcnt/100)
			X_train = data_array[:test_split]
			y_train = sectors_list[:test_split]

			X_test = data_array[test_split:]
			y_test = sectors_list[test_split:]

			split_pcnt =25
			test_split =int(data_array.shape[0]*split_pcnt/100)
			X_train2 = data_array[test_split:]
			y_train2 = sectors_list[test_split:]

			X_test2 = data_array[:test_split]
			y_test2 = sectors_list[:test_split]

			# ## Printing dimensions

			lb = LabelBinarizer()
			y_train_onehot = lb.fit_transform(y_train)
			y_test_onehot = lb.transform(y_test)

			y_train_onehot2 = lb.fit_transform(y_train2)
			y_test_onehot2 = lb.transform(y_test2)

			num_classes = y_train_onehot.shape[1]
			print('num_classes:',num_classes)
			num_features = X_train.shape[1]
			print('num_features:',num_features)
			num_output = y_train_onehot.shape[1]

			try:
				model = tf.keras.models.load_model(models_file+'_'+str(k-1)+'.h5')
				print("loaded previous model")
			except:
				model = create_model(num_features, num_classes)
				model.summary()
				print("created new model")

			checkpoint_path = main_dir+"/cp.ckpt"
			checkpoint_dir = os.path.dirname(checkpoint_path)

			# Create checkpoint callback
			cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
			                                                 save_weights_only=True,
			                                                 verbose=0)

			print("Fold 1")
			model.fit(X_train, y_train_onehot,  epochs = epochs,
			          validation_data = (X_test,y_test_onehot))
			          # ,callbacks = [cp_callback])  # pass callback to training

			print("Fold 2")
			model.fit(X_train2, y_train_onehot2,  epochs = epochs,
			          validation_data = (X_test2,y_test_onehot2))
			          # ,callbacks = [cp_callback])  # pass callback to training

			model.save(models_file+'_'+str(k)+'.h5')

			vector_rep = model.get_weights()[0]


			print(vector_rep.shape)
			print(lb.classes_)
			print(y_train)

			x_to_predict = data_array
			y_predicted = model.predict_proba(x_to_predict)

			print(y_predicted)
			print(y_predicted.shape)

			del model

			# print(names.shape)

			pca = PCA(n_components=2)
			principalComponents = pca.fit_transform(y_predicted)
			principalDf = pd.DataFrame(data = principalComponents)
			predicted_array = (principalDf.values)
			names_array = np.asarray(names)
			names_df = pd.DataFrame(names)
			# print(predicted_array[:,:1])

			# weights_file = 'data_arrays/pca_weights_' +str(k) +'_'+last_date+'.pkl'
			weights_file = main_dir + '/pca_weights_' +str(k) +'.pkl'
			principalDf.to_pickle(weights_file)
			k = k + 1
		with open(main_dir+'/title_list.pkl', 'wb') as f:
			pickle.dump(title_list,f)
	else:
		print("not running model")


	a = AnimatedScatter(columns,total_frames,np.arange(0,99,10),main_dir+'/pca_weights_', False)
	plt.show()


