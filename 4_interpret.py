""""
The aim of the script is to interpret classification model - its performance and feature importances

The script:

1. read best models
2. calculates performance metrics
4. calculates feature importances
3. visualize roc and precision-recall curve
4. Output:
    1. final_model_metrics.csv
    2. feat_imp_[model_name].csv

"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition, manifold
import ast
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
from sklearn.metrics import roc_curve
import plotly
from sklearn.metrics import precision_recall_curve, auc
from sklearn import inspection
import shap
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

########################################################################################################################
"""workspace preparation"""

#import data
df = pd.read_csv('train_data_clean', index_col='match')
#df = df.loc[:,~df.columns.str.contains('^goals_', case=False)]
X = df.loc[:, df.columns != 'win']
y = df['win']

rand = 9

#import best models product of the 3_train.py script
final_model = pd.read_csv('final_model', index_col = 0)

final_model['name_index '] = final_model['Name_m'] + final_model.index.astype(str)

#todo local intrpretation: shap

#if the dataset has balanced target variable?
y.sum()/len(y)

#No Information Rate (NIR)
print('NIR: %.4f' %(y[y==1].shape[0]/y.shape[0]))

########################################################################################################################
"""calculate performance metrics for the best models from 3_train.py"""

pred_y_train = []
prob_y_test = []
pred_y_test = []
acc_tr = []
acc_te = []
rec_tr = []
rec_te = []
roc_te = []
f1_tr = []
mcc_tr = []
explainer_list = []
coeff_imp_df = {}
tree_imp_df = {}
per_imp_df = {}
feature_importances = {}


for row in final_model.itertuples():
    fin_param = ast.literal_eval(row[7])
    if row[1] == 'SVC':
        model = SVC(**fin_param)
    elif row[1] == 'DT':
        model = DecisionTreeClassifier(**fin_param)
    elif row[1] == 'RF':
        model = RandomForestClassifier(**fin_param)
    elif row[1] == 'LR':
        model = LogisticRegression(**fin_param)
    elif row[1] == 'GB':
        model = GradientBoostingClassifier(**fin_param)
    else:
        pass
    X_train, X_test, y_train, y_test = train_test_split(X.loc[:, ast.literal_eval(row[4])], y, test_size=0.33, random_state=rand)
    #X_train = X_train.loc[:, ast.literal_eval(row[4])]
    fitted_model = model.fit(X_train, y_train)
    y_train_pred = fitted_model.predict(X_train.values)
    y_test_prob = fitted_model.predict_proba(X_test.values)[:, 1]
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    Accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
    Accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
    Recall_train = metrics.recall_score(y_train, y_train_pred)
    Recall_test = metrics.recall_score(y_test, y_test_pred)
    ROC_AUC_test = metrics.roc_auc_score(y_test, y_test_prob)
    F1_test = metrics.f1_score(y_test, y_test_pred)
    MCC_test = metrics.matthews_corrcoef(y_test, y_test_pred)
    acc_tr.append(Accuracy_train)
    acc_te.append(Accuracy_test)
    rec_tr.append(Recall_train)
    rec_te.append(Recall_test)
    roc_te.append(ROC_AUC_test)
    f1_tr.append(F1_test)
    mcc_tr.append(MCC_test)
    pred_y_train.append(y_train_pred)
    prob_y_test.append(y_test_prob)
    pred_y_test.append(y_test_pred)
    explainer = ClassifierExplainer(fitted_model, X_test, y_test)
    explainer_list.append(explainer)
    #calculate permutation feature importances
    permutation_imp = inspection.permutation_importance(fitted_model, X_test, y_test, n_jobs=-1,scoring='accuracy', n_repeats=8, random_state=rand)
    permutation_imp_avg = permutation_imp.importances_mean
    # per_imp =  pd.DataFrame({'name': X_test.columns, 'per_feat_imp': permutation_imp_avg})
    # per_imp_df[row] = per_imp
    #calculate feature model intrinsic importances
    if (row[1] == 'SVC' or row[1] == 'LR'):
        # coefficients =  pd.DataFrame({'name': X_train.columns, 'coef_norm': fitted_model.coef_[0]})
        # coeff_imp_df[row] = coefficients
        shap_tree_based_explainer = shap.KernelExplainer(fitted_model.predict_proba)
        shap_tree_values_train = shap_tree_based_explainer.shap_values(X_train)[1]
        shap_tree_values_test = shap_tree_based_explainer.shap_values(X_test)[1]
        shap_tree_values_train_avg = np.mean(shap_tree_values_train, axis=0)
        shap_tree_values_test_avg = np.mean(shap_tree_values_test, axis=0)
        feat_imp = pd.DataFrame({'name': X_train.columns,
                                                    'int_feat_imp': fitted_model.coef_[0], 'permu_imp' : permutation_imp_avg,
                                                    'shap_train': shap_tree_values_train_avg, 'shap_test': shap_tree_values_test_avg})
        # shap_tree_values_test_avg =  pd.DataFrame({'name': X_train.columns, 'tree_feat_shap': shap_tree_values_test_avg})
        feature_importances[row[8]] = feat_imp
    elif (row[1] == 'DT' or row[1] == 'RF' or row[1] == 'GB'):
        # tree_based_imp =  pd.DataFrame({'name': X_train.columns, 'tree_feat_imp': fitted_model.feature_importances_})
        # tree_imp_df[row] = tree_based_imp
        shap_tree_based_explainer = shap.TreeExplainer(fitted_model)
        shap_tree_values_train = shap_tree_based_explainer.shap_values(X_train)[1]
        shap_tree_values_test = shap_tree_based_explainer.shap_values(X_test)[1]
        shap_tree_values_train_avg = np.mean(shap_tree_values_train, axis=0)
        shap_tree_values_test_avg = np.mean(shap_tree_values_test, axis=0)
        feat_imp = pd.DataFrame({'name': X_train.columns,
                                                    'int_feat_imp': fitted_model.feature_importances_, 'permu_imp' : permutation_imp_avg,
                                                    'shap_train': shap_tree_values_train_avg, 'shap_test': shap_tree_values_test_avg})
        # shap_tree_values_test_avg =  pd.DataFrame({'name': X_train.columns, 'tree_feat_shap': shap_tree_values_test_avg})
        feature_importances[row[8]] = feat_imp
    else:
        pass


final_model['pred_y_train'] = pred_y_train
final_model['acc_tr'] = acc_tr
final_model['acc_te'] = acc_te
final_model['rec_tr'] = rec_tr
final_model['rec_te '] = rec_te
final_model['roc_te'] = roc_te
final_model['f1_tr'] = f1_tr
final_model['mcc_tr'] = mcc_tr
final_model['explainer'] = explainer_list

final_model.to_csv('final_model_metrics')


final_model_metrics = pd.read_csv('final_model_metrics', index_col = 0)

for name, df in feature_importances.items():
    file_name = "feat_imp_" + name + ".csv"
    df.to_csv(file_name)



 file_name = row['Name']+".csv"  #Change the column name accordingly
    pd.DataFrame(row).T.to_csv(file_name, index=None)
########################################################################################################################
"""visualize the roc and auc"""

#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()

plotly.offline.plot(fig, filename='file.html')


#Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()
plotly.offline.plot(fig, filename='file.html')



















# ridge: Ridge classification leverages the same regularization technique used
# in ridge regression but applied to classification. It does this by converting the
# target values to -1 (for a negative class) and keeping 1 for a positive class and then
# performing ridge regression. At its heart, its regression in disguise will predict
# values between -1 and 1, and then convert them back to a 0-1 scale. Like with
# RidgeCV for regression, RidgeClassifierCV uses leave-one-out crossvalidation,
# which means it first splits the data into different equal-size sets – in
# this case, we are using five sets (cv=5) – and then removes features one at a time
# to see how well the model performs without them, on average in all the five sets.
# Those features that don't make much of a difference are penalized testing several
# regularization strengths (alphas) to find the optimal strength. As with all
# regularization techniques, the point is to discourage learning from unnecessary
# complexity, minimizing the impact of less salient features.

# knn: kNN can also be applied to classification tasks, except instead of averaging
# what the nearest neighbors' target features (or labels) are, it chooses the most
# frequent one (also known as the mode). We are also using a of 7 for classification
# (n_neighbors).

# naive_bayes: Gaussian Naïve Bayes is part of the family of Naïve Bayes
# classifiers, which are called naïve because they make some assumptions that
# the features are independent of each other, which is usually not the case. This
# dramatically impedes its capacity to predict unless the assumption is correct. It's
# called Bayes because it's based on Bayes' theorem of conditional probabilities,
# which is that the conditional probability of a class is the class probability times
# the feature probability given the class. Gaussian Naïve Bayes makes an additional
# assumption, which is that continuous values have a normal distribution, also known
# as a Gaussian distribution.

# gradient_boosting: Like random forest, gradient boosted trees are also
# an ensemble method, but that leverages boosting instead of bagging. Boosting
# doesn't work in parallel but in sequence, iteratively training weak learners and
# incorporating their strengths into a stronger learner, while adapting another weak
# learner to tackle their weaknesses. Although ensembles and boosting, in particular,
# can be done with a model class, this one uses decision trees. We have limited the
# number of trees to 210 (n_estimators=210).

# mlp: The same multi-layer perceptron as with regression, but the output layer,
# by default, uses a logistic function in the output layer to yield probabilities, which
# it then converts to 1 or 0 , based on the 0.5 threshold. Another difference is that
# we are using seven neurons in the first and only hidden layer (hidden_layer_
# sizes=(7,)) because binary classification tends to require fewer of them to
# achieve an optimal result.

# Accuracy: Accuracy is the simplest way to measure the effectiveness of a
# classification task, and it's the percentage of correct predictions over all predictions.
# In other words, in a binary classification task, you can calculate this by adding the
# number of True Positives (TPs) and True Negatives (TNs) and dividing them by a
# tally of all predictions made. As with regression metrics, you can measure accuracy
# for both train and test to gauge overfitting.
# • Recall: Even though accuracy sounds like a great metric, recall is much better in
# this case and the reason is you could have an accuracy of 94%, which sounds pretty
# good, but it turns out you are always predicting no delay! In other words, even if
# you get high accuracy, it is meaningless unless you are predicting accurately for the
# least represented class, delays. We can find this number with recall (also known as
# sensitivity or true positive rate), which is TP / TP + FN and it can be interpreted as
# how much of the relevant results were returned. In other words, in this case, what
# percentage of the actual delays were predicted. Another good measure involving
# true positives is precision, which is how much our predicted samples are relevant,
# which is TP / TP + FP. In this case, that would be what percentage of predicted
# delays were actual delays. For imbalanced classes, it is recommended to use both,
# but depending on your preference for over , you will prefer recall over
# precision or vice versa.

# F1: The F1-score is also called the harmonic average of precision and recall because
# it's calculated like this: 2TP / 2TP + FP + FN. Since it includes both precision and
# recall metrics, which pertain to the proportion of true positives, it's a good metric
# choice to use when your dataset is imbalanced, and you don't prefer either precision
# or recall.
# • MCC: The Matthews correlation coefficient is a metric drawn from biostatistics.
# It's gaining popularity in the broader data science community because it has the
# ability to produce high scores considering TP, FN, TN, and FP fairly because it
# takes into account proportions of classes. This makes it optimal for imbalanced
# classification tasks. Unlike all other metrics used so far, it doesn't range from 0 to
# 1 but -1, complete disagreement, to 1, a total agreement between predictions and
# actuals. The mid-point, 0, is equivalent to a random prediction.


# ROC-AUC: ROC is an acronym for Receiver Operating Characteristic and was
# designed to separate signal from noise. What it does is plot the proportion of true
# positive rate (Recall) on the x axis and the false positive rate on the y axis. AUC
# stands for area under the curve, which is a number between 0 and 1 that assesses
# the prediction ability of the classifier 1 being perfect, 0.5 being as good as a coin
# toss, and anything lower meaning that if we inverted the results of our prediction,
# we would have a better prediction. To illustrate this, let's generate a ROC curve for
# our worse-performing model, Naïve Bayes, according to the AUC metric:

########################################################################################################################
#dimensionality reduction methods

# of dimensionality reduction, and it's usually done by performing eigenvalue
# decomposition of the covariance matrix of the data. Unlike the others we
# are exploring here, it's computationally speedy. The process of eigenvalue
# decomposition finds orthogonal vectors, which means that geometrically they are
# far apart. This is so that PCA can reduce dimensions to ones that are uncorrelated
# to each other. Its name refers to principal components because eigenvectors are also
# called principal directions. This makes sense because data is reduced by projecting
# data to fewer dimensions while trying not to lose information, so it assumes
# directions with the greatest variances are the most important.
# • t-sne: T-distributed Stochastic Neighbor Embedding (t-SNE) is one of the
# newer methods of dimensionality reduction, and unlike PCA, it is non-linear, so it's
# good at capturing non-linearities. Also unlike PCA, the mathematical theory behind
# t-SNE is not linear algebra but probability. It minimizes the difference between
# pairwise distribution similarities between high-dimensional (our input data) and
# the lower-dimensional representation using Kullback-Leibler divergence (which is
# a distance measurement). Unlike PCA, which focuses on putting dissimilar points
# as far apart as possible, t-SNE is about placing similar points close together.

dimred_methods = {
        #Decomposition
        'pca':{'method': decomposition.PCA(n_components=3, random_state=rand)},
        #Manifold Learning
        't-sne':{'method': manifold.TSNE(n_components=3, random_state=rand)},
    }

for method_name in dimred_methods.keys():
    lowdim_data = dimred_methods[method_name]['method'].fit_transform(X_test.values)
    dimred_methods[method_name]['lowdim'] = lowdim_data


features = X_test.columns

fig = px.scatter_matrix(
    X_test,
    dimensions=features,
    color= y_test_pred
)
fig.update_traces(diagonal_visible=False)
fig.show()
plotly.offline.plot(fig, filename='file.html')


import plotly.express as px
from sklearn.decomposition import PCA

features = X.columns

pca = PCA()
components = pca.fit_transform(df[features])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=y,
    dimensions=range(4),
    color=y
)
fig.update_traces(diagonal_visible=False)
fig.show()
plotly.offline.plot(fig, filename='file.html')