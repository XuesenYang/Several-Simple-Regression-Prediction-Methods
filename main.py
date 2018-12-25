import pandas as pd
import numpy as np

##load data
df_train = pd.read_csv('zhengqi_train.txt',sep='\t')
df_test = pd.read_csv('zhengqi_test.txt',sep='\t')
train_x=df_train.drop(['target'],axis=1)
all_data = pd.concat([train_x,df_test])


##data visualization
import seaborn
import matplotlib.pyplot as plt

for col in all_data.columns:
    seaborn.distplot(train_x[col])
    seaborn.distplot(df_test[col])
    plt.show()

## 'V5', 'V17', 'V28', 'V22', 'V11', 'V9' could be deleted
all_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1,inplace=True)

## data standardization--method--Min-maxnormalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_scaler.fit_transform(all_data),columns=all_data.columns)

## skew-revision---to fit linear model
import math

data_minmax['V0'] = data_minmax['V0'].apply(lambda x:math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x:math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x:math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])
X_scaled = pd.DataFrame(preprocessing.scale(data_minmax),columns = data_minmax.columns)
train_x = X_scaled.ix[0:len(df_train)-1]
test = X_scaled.ix[len(df_train):]
Y=df_train['target']

## feature selection---Through the variance threshold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

threshold = 0.85
vt = VarianceThreshold().fit(train_x)
feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1-threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]

## single feature---Select features according to the k highest scores.
# see detail -- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
X_scored = SelectKBest(score_func=f_regression, k=10).fit(train_x, Y) # F-value between label/feature for regression tasks.
print(X_scored)
feature_scoring = pd.DataFrame({'feature': train_x.columns,'score': X_scored.scores_})
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)

## selected model
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

n_folds = 10

def rmsle_cv(model,train_x_head=train_x_head):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    rmse= -cross_val_score(model, train_x_head, Y, scoring="neg_mean_squared_error", cv = kf)
    return(rmse)

## SVR
svr = make_pipeline( SVR(kernel='linear'))
## LR
line = make_pipeline( LinearRegression())
## Lasso
lasso = make_pipeline( Lasso(alpha =0.0005, random_state=1))
## ElasticNet
ENet = make_pipeline( ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
## KernelRidge(3 versions)
KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
KRR3 = KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
## Gradient boosting(GB)
GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.02,
                                    max_depth=5, max_features=7,
                                    min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
## Xgboost
model_xgb = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.8, gamma=0.1,
                             learning_rate=0.02, max_depth=5,
                             n_estimators=500,min_child_weight=0.8,
                             reg_alpha=0, reg_lambda=1,
                             subsample=0.8, silent=1,
                             random_state =42, nthread = 2)

## lightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                               learning_rate=0.05, n_estimators=720,
                               max_bin = 55, bagging_fraction = 0.8,
                               bagging_freq = 5, feature_fraction = 0.2319,
                               feature_fraction_seed=9, bagging_seed=9,
                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
## Random Forest(RF)
rf = RandomForestRegressor(n_estimators= 50, max_depth=25, min_samples_split=20,
                                  min_samples_leaf=10,max_features='sqrt' ,oob_score=True, random_state=10)
## print result
score = rmsle_cv(svr)
print("\nSVR performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(line)
print("\nLine performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(lasso)
print("\nLasso performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR1)
print("Kernel Ridge1 performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR2)
print("Kernel Ridge2 performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR3)
print("Kernel Ridge3 performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
GBoost.fit(train_x_head,Y)
print("Gradient Boosting performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost performance: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(train_x_head,Y)

score = rmsle_cv(model_lgb)
print("LGBM performance: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

score = rmsle_cv(rf)
print("RF performance: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

## Predict test set and save results
final = pd.DataFrame()
df_test.drop(feat_scored_headnum,axis=1,inplace=True)
final['SVR'] = svr.predict(df_test)
final['Line'] = line.predict(df_test)
final['Lasso'] = lasso.predict(df_test)
final['ElasticNet'] = ENet.predict(df_test)
final['Kernel Ridge1'] = KRR1.predict(df_test)
final['Kernel Ridge2'] = KRR2.predict(df_test)
final['Kernel Ridge3'] = KRR3.predict(df_test)
final['Gradient Boosting'] = GBoost.predict(df_test)
final['Xgboost'] = model_xgb.predict(df_test)
final['lightgbm'] = model_lgb.predict(df_test)
final['RF'] = rf.predict(df_test)
np.savetxt('final.txt',final)
print('File saved!')






