import pandas as pd 
import numpy as np 
from scipy import stats
from data_processing import *
from os import listdir
from os.path import isfile, join



from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble  import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier


if __name__ == '__main__':


	x , y , num , cat = data_prep()

	classifiers = [
			('KNN' , KNeighborsClassifier() , {
												'KNN__algorithm':('auto','ball_tree', 'kd_tree', 'brute'),
										        'KNN__n_neighbors' : stats.randint(4 ,40),
										        'KNN__p' : stats.randint(1 ,5),
										        'KNN__weights' : ('uniform' , 'distance')
				}),
			('ETclf' , ExtraTreesClassifier() , {
	                            'ETclf__n_estimators': stats.randint(10 ,500),
	                            'ETclf__criterion' :('gini' , 'entropy'), 
	                            'ETclf__max_depth': stats.randint(10 ,50), 
	                            'ETclf__max_features': ('sqrt' , 'log2', 'auto') , 
	#                             'ETclf__max_leaf_nodes' :(None , 2 , 4 , 5 , 6 , 7 , 9 , 10) ,  
	#                             'ETclf__min_samples_leaf':[2, 4 , 6 , 8 , 10], 
	#                             'ETclf__min_samples_split':[2, 4 , 6 , 8 , 10]
	                                            }), 
			('SGD' , SGDClassifier() , {
						'SGD__loss' : ('hinge' , 'log'),
						'SGD__penalty' : ('l1' , 'l2' , 'elasticnet'),
						'SGD__alpha' : stats.uniform(loc = 0 , scale = 0.5)
				} )
	]


	pipe = Pipeline(steps=[

		('handle_MV' , Impute_MV()),

		('union' , FeatureUnion([
			('numerical' , Pipeline(steps = [
				('extract' , ColumnExtractor(num)),
				('scale' , StandardScaler())
				])),
			('categorical' , Pipeline(steps = [
				('extract' , ColumnExtractor(cat)),
				('one_hot' , One_Hot())
				]))
				

			]))


		])

	onlyfiles = [f for f in listdir('data') if isfile(join('data', f))]

	for name , clf , params in classifiers : 

		url = f'data/{name}_model_selection_algo.csv'
		if url[5:] not in onlyfiles : 
			pipe.steps.append((name , clf ))

			cv = StratifiedKFold(5)
			grid = RandomizedSearchCV(pipe , params , cv = cv , n_jobs = -1 , scoring = 'accuracy' )
			grid.fit(x , y)
			means = grid.cv_results_['mean_test_score']
			stds = grid.cv_results_['std_test_score']
			params = grid.cv_results_['params']
			df = pd.DataFrame({
				    'means' : means,
				    'stds' : stds,
				    'params' : params
			})

			df.to_csv(url)
			pipe.steps = pipe.steps[:-1]
		else : 
			continue
