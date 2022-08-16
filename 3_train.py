""""
The aim of the script is to find the best classification method

The script:

1. train models with k best feature selection
2. train models with rfecv
3. search for best hyperparameters for best 3 models
4. Output:
    1. final_model.csv

"""





# import libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif


########################################################################################################################
"""Models - data import and variables preparation"""

# import data
df = pd.read_csv('train_data_clean', index_col='match')
# df = df.loc[:,~df.columns.str.contains('^goals_', case=False)]
X = df.loc[:, df.columns != 'win']
y = df['win']

#definition of cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#models we check
models = []
models.append(('SVC', SVC())) #linear
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier())) #treebased
models.append(('RF', RandomForestClassifier())) #treebased
models.append(('GB', GradientBoostingClassifier())) #treebased


#variables to store the modelling results
names = []
scores = []
name_e = []
n_feat = []
model_name = []

########################################################################################################################
"""Models - filter model selection - SelectKBest"""

fs = SelectKBest(score_func=mutual_info_classif)

for name_m, model in models:
	pipeline = Pipeline(steps=[
		('select', SelectKBest(score_func=mutual_info_classif)),
		('classify', model)])
	grid = dict()
	grid['select__k'] = [i + 1 for i in range(X.shape[1])]
	cv_pipeline = GridSearchCV(estimator=pipeline, param_grid=grid)
	results = cv_pipeline.fit(X=X, y=y)
	final_pipeline = cv_pipeline.best_estimator_
	final_classifier = final_pipeline.named_steps['classify']
	select_indices = final_pipeline.named_steps['select'].get_support(indices=True)
	feature_names = [X.columns[i] for i in select_indices]
	print('Best Mean Accuracy: %.3f' % cv_pipeline.best_score_)
	names.append(name_m)
	scores.append(results.best_score_)
	name_e.append(fs)
	n_feat.append(feature_names)
	model_name.append(model)

########################################################################################################################
"""Models - RFCV model selection"""

estimators = []
estimators.append(('LR', LogisticRegression()))
estimators.append(('DT', DecisionTreeClassifier()))
estimators.append(('RF', RandomForestClassifier()))
estimators.append(('GB', GradientBoostingClassifier()))

for name, esti in estimators:
	rfecv = RFECV(estimator=esti, step=1, min_features_to_select=10, cv=10, scoring='accuracy', verbose=0, n_jobs=None)
	rfecv.fit(X, y)
	feature_importance = list(zip(X.columns, rfecv.support_))
	new_features = []
	for key, value in enumerate(feature_importance):
		if (value[1]) == True:
			new_features.append(value[0])
	X_new = X[new_features]
	for name_m, model in models:
		score = cross_val_score(model, X_new, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
		names.append(name_m)
		scores.append(score)
		name_e.append(name)
		n_feat.append(new_features)
		model_name.append(model)

########################################################################################################################
"""Models - results of kBest and RFCV models stored in df"""

kf_cross_val = pd.DataFrame({'Name_m': names, 'Name_e': name_e, 'Score': scores, 'feat' : n_feat, 'model': model_name})

########################################################################################################################
"""Models hyperparameters tuning for 3 best"""

m = kf_cross_val.nlargest(3,'Score')

grid_score = []
best_param = []

for row in m.itertuples():
	if row[1] == 'SVC':
		model = SVC()
		kernel = ['poly', 'rbf', 'sigmoid', 'linear']
		C = [50, 10, 1.0, 0.1, 0.01]
		gamma = ['scale']
		# define grid search
		grid = dict(kernel=kernel, C=C, gamma=gamma)
	elif row[1] == 'DT':
		model = DecisionTreeClassifier()
		max_depth = [3, None]
		max_features = ['auto', None]
		min_samples_leaf = [1,2,3,4]
		criterion = ["gini", "entropy"]
		# define random search
		grid = dict(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf,
					criterion=criterion)
	elif row[1] == 'RF':
		model = RandomForestClassifier()
		n_estimators = [8,9,10,11,12]
		# Number of features to consider at every split
		max_features = ['auto', 'sqrt']
		# Maximum number of levels in tree
		max_depth = [3, None]
		# Minimum number of samples required to split a node
		min_samples_split = [2, 5, 10]
		# Minimum number of samples required at each leaf node
		min_samples_leaf = [1, 2, 3, 4]
		# Method of selecting samples for training each tree
		bootstrap = [True, False]
		# Create the random grid
		grid = {'n_estimators': n_estimators,
				'max_features': max_features,
				'max_depth': max_depth,
				'min_samples_split': min_samples_split,
				'min_samples_leaf': min_samples_leaf,
				'bootstrap': bootstrap}
	elif row[1] == 'LR':
		model = LogisticRegression()
		solvers = ['newton-cg', 'lbfgs', 'liblinear']
		penalty = ['l2']
		c_values = [100, 10, 1.0, 0.1, 0.01]
		# define grid search
		grid = dict(solver=solvers, penalty=penalty, C=c_values)
	elif row[1] == 'GB':
		model = GradientBoostingClassifier()
		n_estimators = [10, 100, 1000]
		learning_rate = [0.001, 0.01, 0.1]
		subsample = [0.5, 0.7, 1.0]
		max_depth = [3, 7, 9]
		# define grid search
		grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
	else:
		pass
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
								   error_score=0)
	X = df.loc[:, row[4]]
	grid_result = grid_search.fit(X, y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_), row[1], row[2])
	grid_score.append(grid_result.best_score_)
	best_param.append(grid_result.best_params_)
	grid_result(grid_result.best_estimator_)
m['grid_score'] = grid_score
m['best_param'] = best_param

m.to_csv('final_model')



#
# final_model = pd.read_csv('final_model', index_col = 0)
# fin_param = ast.literal_eval(final_model.loc[18,'best_param'])
# model_fin = RandomForestClassifier(**fin_param)
#
# X_fin = df.loc[:, ast.literal_eval(final_model.iloc[0,3])]
#
# model_fin.fit(X_fin, y)
#
# for name, score in zip(X_fin.columns, model_fin.feature_importances_):
# 	print(name, score)







