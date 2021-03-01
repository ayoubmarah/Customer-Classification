import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def data_prep():
	X = pd.read_csv('data/Train_set.csv')
	X.drop('ID' , axis = 1 , inplace = True)
	numerical = []
	categorical = []

	for i in X.columns : 
	
		if X[i].dtype == np.object and i != 'Segmentation' :
			categorical.append(i)
		elif i != 'Segmentation' : 
			numerical.append(i)
	
	X.drop('Work_Experience' , axis = 1 , inplace = True)


	for i in X.columns : 
	
		if X[i].isna().sum() and  X[i].dtype == 'O':		
			X[i] = X[[i]].apply(lambda x:x.fillna(x.value_counts().index[0]))
		
		elif X[i].isna().sum(): 
			X[i] = X[[i]].fillna(X[i].median())



	for col in categorical:
		dummies = pd.get_dummies(X[col], drop_first=False)
	
		dummies = dummies.add_prefix('{}_'.format(col))
		
		X.drop(col, axis=1, inplace=True)
		
		X = X.join(dummies)

	
	a = X.Segmentation
	le = LabelEncoder()
	le.fit(a)
	X['Segmentation'] = le.fit_transform(a)

	y = X['Segmentation']

	x = X[[i for i in X.columns if i != 'Segmentation']]

	return x , y 


def proxmatrix(model , X , normalize = True):

	

	"""

	Create the proximity matrix that can be done to fill missing values based on their similarity and also

	use the matrix for clustering problems 

	model.apply(X) : Returns

	X_leavesndarray of shape (n_samples, n_estimators)

	For each datapoint x in X and for each tree in the forest, return the index of the leaf x ends up in
"""

	terminals = model.apply(x)

	nTrees = terminals.shape[1]


	a = terminals[:,0]

	proxMat = 1*np.equal.outer(a, a)

	for i in range(1 , nTrees):
		a = terminals[:,i]
		proxMat += 1*np.equal.outer(a, a)

	
	if normalize :
		proxMat = proxMat / nTrees

	return proxMat


def imputing(X , col  , proximity_matrix):
	
	nr , nc = X.shape
	O_i = X[col][X[col].isna() == False]
	M_i = X[col][X[col].isna() == True ]
	n_oi = O_i.shape[0]
	n_mi = M_i.shape[0]
	if X[col].dtype == 'O':
		for i in M_i.index:
			
			# list of weighted frequency of each label 
			weighted_f = []
			# iterate through labels
			for j in X[col].unique() : 
				# get a sub Serie contain jth label 
				label = X[col][X[col] == j]
				n_label = label.shape[0]
				# calculate label frequency 
				f_label = n_label / n_oi
				# calculate proximity of a label 
				proximity =  np.sum(proximity_matrix[ i , label.index ]) / np.sum(proximity_matrix[ i , : ])
				# calculate the weighted frequency of a label 
				weighted_f.append(f_label * proximity)
			
			approx_val = X[col].unique()[weighted_f.index(max(weighted_f))]
			X[col][i] = approx_val
	else : 
		for i in M_i.index : 
			value = 0
			for j in O_i.index:
				# calculate the weighted average
				value +=  O_i[j] * ( proximity_matrix[ j , i ] / np.sum(proximity_matrix[ i , : ])  )
				
			X[col][i] = value
				
	return X


if __name__ == '__main__' : 

	x , y = data_prep()
	
	RFclf = RandomForestClassifier()
	best_params_ = {'max_depth': 11, 'max_features': 'log2', 'n_estimators': 406}
	model = RFclf.set_params(**best_params_)
	
	for i in range(5) : 

		model.fit(x,y)
			
		prox_matrix = proxmatrix(model , x)
		for col in x.columns:
			imputing(x , col , prox_matrix )
			
	