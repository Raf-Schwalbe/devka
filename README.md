# Devka

Purposes of the project are: 
- scrap data fot the analysis
- build a classification model that is able to classify whether match is won or not
- make and deploy dashboard

The code is supposed to work also for other teams listed in playarena

## Table of contents

### 1. get the data 
- Aim of this script is to get data for the analysis from the web site.

It is done in following way:

1. selenium opens website (if you want to make analysis for other team you need to change link in driver.get
Next it accept cookies and load ajax to have full list of meetings. Then it saves links to meetings

2. the links to meetings are used by beautifulsoup which opens every meeting and scrap it

3. output:

- whole data: 'raw_data.csv'
- matches data: 'raw_data_matches.csv'
- players data: 'raw_data_players.csv'


### 2. prepare the data

The aim of the script is to prepare data for training

The script:

1. cleans and recodes data
2. creates new features
3. imputes missing data
4. remove uninformative features
5. makes dummmy variables
6. scales variables
7. Output:
    1. players_data_clean.csv
    2. match_data_clean.csv (there are made points 1-2 only)
    3. train_data_clean.csv

### EDA and classic classification methods (main)

- Introduction - what do we want to achieve?
- Exploratory Data Analysis
  - Variation
    - Outliers
    - Only 0 observations
  - Covariation
- Modelling
  - Models - filter model selection - SelectKBest
  - Models - RFCV model selection
  - Model results
  - Hyperparameters tuning for top model.
- Best model Interpretation
  - Performance
  - Metrics
  - Validation curves
  - Independent variables p-value check
  - Feature importance
- Summary

### Neural network (main_nn)

- Introduction
- Modelling 
  - Hyperparameters tunning
- Interpretation
- Summary
- Scikit learn w Keras


## Interesting features

- cross validation where estimator (for RFECV) and classification algorithm) are assesed.


## Results

- The best model is Logisitc regression with features selected with Logistic regression. Its mean accuracy was 82,63%

- In general:

  - tree based models performed visibly worse
  - the estimator used in RFCV did not really matter  unless it was decision tree which worsened model performance
  - models with feature selected with filter method performs worse than with RFCV unless they are tree based.
