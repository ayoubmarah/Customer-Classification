# import numpy as np
# import pandas as pd 
# from sklearn.base import TransformerMixin

# def data_prep():
# 	X = pd.read_csv('data/Train_set.csv')
# 	X.drop('ID' , axis = 1 , inplace = True)
# 	numerical = []
# 	categorical = []
# 	X.drop('Work_Experience' , axis = 1 , inplace = True)



# 	for i in X.columns : 
	
# 		if X[i].dtype == np.object and i != 'Segmentation' :
# 			categorical.append(i)
# 		elif i != 'Segmentation' : 
# 			numerical.append(i)
	
	
# 	y = X['Segmentation']

# 	x = X[[i for i in X.columns if i != 'Segmentation']]

# 	return x , y  , numerical , categorical 
	
# class ColumnExtractor (TransformerMixin):
# 	def __init__(self , features):
# 		self.features = features

# 	def fit(self , X , y = None):
# 		return self
# 	def transform(self , X , y = None ):
# 		X_ = X.copy()
# 		return X_[self.features]
		
# class Impute_MV(TransformerMixin):
# 	def __init__(self):
# 		pass
# 	def fit(self , X , y = None ):
# 		return self
# 	def transform(self , X , y = None):
# 		X_ = X.copy()
# 		for i in X_.columns : 
# 			if X_[i].isna().sum() and  X_[i].dtype == 'O':
# 				X_[i] = X_[[i]].apply(lambda x:x.fillna(x.value_counts().index[0]))
# 			elif X_[i].isna().sum(): 
# 				X_[i] = X_[[i]].fillna(X_[i].mean())
# 		return X_

# class One_Hot(TransformerMixin):
# 	def __init__(self):
# 		pass
# 	def fit(self , X , y=None):
# 		return self 
# 	def transform(self , X , y=None):
# 		X_ = X.copy()
# 		categorical = X_.columns
# 		for col in categorical:
# 			dummies = pd.get_dummies(X_[col], drop_first=False)
# 			dummies = dummies.add_prefix('{}_'.format(col))
# 			X_.drop(col, axis=1, inplace=True)
# 			X_ = X_.join(dummies)
# 		return X_













import sys 

print('hello world')
# print(sys.version)
print(sys.executable)






