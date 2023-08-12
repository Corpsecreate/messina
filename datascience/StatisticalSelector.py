import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sklearn.metrics
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\samsg\Downloads\loan_approval_dataset.csv"); df['label'] = (df[' loan_status'] == " Approved")
#df = pd.read_csv(r"C:\Users\samsg\Downloads\uci-secom.csv"); df['label'] = df['Pass/Fail'] == 1
df = pd.read_csv(r"C:\Users\samsg\Downloads\Property Sales of Melbourne City.csv")

#df = df.sample(frac=0.5, random_state = 1)
X = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN']
Y = "AGE"
X = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
Y = 'Class'
X = ['loan_id', ' no_of_dependents', ' income_annum', ' loan_amount',
       ' loan_term', ' cibil_score', ' residential_assets_value',
       ' commercial_assets_value', ' luxury_assets_value',
       ' bank_asset_value']
Y = 'label'

dumms = pd.get_dummies(df[['Type', 'Method', 'CouncilArea', 'Regionname']], dtype=bool)
df[dumms.columns] = dumms
X = list(df.select_dtypes('float').columns) + list(dumms.columns)
Y = "Price"

#X     = list(df.select_dtypes('float').columns)
df[X] = df[X].astype(float)
df[X] = df[X].fillna(-99999)
df[Y] = df[Y].astype(float)

binner = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='quantile', subsample=None)

#for x in X:
#    n_x   = len(pd.unique(df[x]))
#    q     = min(1000, n_x)
#    df[x] = pd.qcut(df[x], q, duplicates='drop').cat.codes
    
#df[X]  = binner.fit_transform(df[X])

#SCORER        = "neg_log_loss"
#SCORER        = "neg_brier_score"
SCORER        = "neg_root_mean_squared_error"
cv            = KFold(5, shuffle=True, random_state = 1)

min_var_split = 1e-6 * np.var(df[Y])
#base_score    = np.mean(cross_val_score(DummyClassifier(), df[X], df[Y], cv = cv, n_jobs=1, scoring=SCORER))
base_score    = np.mean(df[Y])

base_model = RandomForestRegressor(n_estimators=4, max_features=0.50, max_samples=0.20, n_jobs=-1, min_impurity_decrease=min_var_split)
base_model = RandomForestRegressor(n_estimators=8, max_features=0.666, max_samples=0.80, n_jobs=-1, min_impurity_decrease=min_var_split)
#base_model = RandomForestClassifier(criterion="gini", n_estimators=16, max_features=0.80, max_samples=0.80, n_jobs=-1, min_impurity_decrease=min_var_split)
#base_model = XGBClassifier(base_score=base_score, eval_metric='logloss', tree_method='hist', max_bin=1000, colsample_bynode=0.666, subsample=0.50, grow_policy='lossguide', learning_rate=0.30, n_jobs=-1)
#base_model = DecisionTreeClassifier(criterion="log_loss", max_features=1.0, splitter="best")
#base_model = DecisionTreeRegressor(min_impurity_decrease=min_var_split)

grid = GridSearchCV(base_model,
                    {"min_samples_leaf" : [1, 2, 3, 4,],},
                    cv=cv, 
                    refit=False, 
                    #scoring="neg_root_mean_squared_error",
                    #scoring="neg_brier_score",
                    scoring=SCORER,
                    n_jobs=-1,
                    verbose=0,
                    )

N_TO_SELECT  = 500
N_RANDOM     = 64
n_yes        = int(N_RANDOM / 2 + 0.5)
n_no         = N_RANDOM - n_yes
SELECTED     = np.zeros(len(X))
selected_x   = []
importances_ = []
scores       = []
method       = 'tval'
alpha        = 0.20
peak_sample  = None

for n_selected in range(min(N_TO_SELECT, len(X))):
    
    flags        = np.repeat([True] * n_yes + [False] * n_no, len(X)).reshape(N_RANDOM, -1)
    rng          = np.random.default_rng()
    flags        = rng.permuted(flags, axis=0)
    perm_results = pd.DataFrame()
    
    for perm_i in tqdm(range(N_RANDOM)):
        
        if perm_i == 0:
            masks = [i % 2 == 0 for i in range(len(X))]
        elif perm_i == 1:
            masks = [i % 2 == 1 for i in range(len(X))]
        else:
            masks = flags[perm_i]
            while sum([x for i, x in enumerate(masks) if SELECTED[i] != 1]) == 0:
                masks = np.random.choice([True, False], len(X))
            
        masks              = [True if SELECTED[i] == 1 else x for i, x in enumerate(masks)]
        random_df          = pd.DataFrame(np.array(masks, dtype=bool).reshape(-1, len(masks)), columns=X)
        sub_x              = [x for i, x in enumerate(X) if masks[i]]
        score              = cross_val_score(base_model, df[sub_x], df[Y], cv = cv, n_jobs=1, scoring=SCORER)
        random_df['score'] = np.mean(score)
        perm_results       = pd.concat([perm_results, random_df], ignore_index=True)
    
    remain_x = [x for i, x in enumerate(X) if SELECTED[i] == 0]
    coefs    = [0] * len(remain_x)
    pvals    = [0] * len(remain_x)
    tstats   = [0] * len(remain_x)
    
    for i, x in enumerate(remain_x):
        when_on     = perm_results[perm_results[x] == True]
        when_off    = perm_results[perm_results[x] == False]
        test_result = stats.ttest_ind(when_on['score'], when_off['score'], equal_var=False, alternative='greater')
        coefs[i]    = when_on['score'].mean() - when_off['score'].mean()
        pvals[i]    = test_result.pvalue
        tstats[i]   = test_result.statistic
        
    #stats_x  = sm.add_constant(perm_results[remain_x]).astype('int')
    #ols      = sm.OLS(perm_results['score'], stats_x).fit()
    #coefs    = [t for t in ols.params[1:]] if method == 'coef' else [t for t in ols.tvalues[1:]]
    max_i    = np.argmax(coefs if method == 'coef' else tstats)
    max_feat = remain_x[max_i]
    
    #print("{:<16}{:<12,.6f}{:<12.6f}".format(max_feat, ols.params[max_feat], ols.pvalues[max_feat]))
    print("{:<16} {:<12,.6f} {:<12.6f}".format(max_feat, coefs[max_i], pvals[max_i]))
    #if ols.params[max_feat] < 0:# or ols.pvalues[max_feat] > alpha:
    #if coefs[max_i] < 0 or pvals[max_i] > alpha:
    #    break
    
    if peak_sample is not None:
        pval = stats.ttest_ind(perm_results['score'], peak_sample, equal_var=False, alternative='greater').pvalue
        print("Score Improvement p < {:.3f}".format(pval))
        if pval > 0.90:
            break

    SELECTED[X.index(max_feat)] = 1
    selected_x.append(max_feat)
    importances_.append(max(coefs))
    scores.append(perm_results['score'])
    if peak_sample is None or (peak_sample is not None and perm_results['score'].mean() >= peak_sample.mean()):
        peak_sample = perm_results['score']
    
importances_norm_ = [x / sum(importances_) for x in importances_]

fig, ax = plt.subplots()
ax.barh(selected_x[::-1], importances_norm_[::-1], color='dodgerblue')
#ax.set_label([f'{p:.1f}%' for p in importances_norm_[::-1]])
ax.bar_label(ax.containers[0], labels=[f'{100*p:.1f}%' for p in importances_norm_[::-1]])
plt.show()

plt.plot([s.mean() for s in scores], '-o')
plt.show()

cv2   = KFold(10, shuffle=True, random_state = 7777)
final = RandomForestClassifier(criterion="gini", n_estimators=32, max_features=0.80, bootstrap=False, n_jobs=8, min_impurity_decrease=min_var_split)

grid = GridSearchCV(base_model,
                    {"min_samples_leaf" : [1, 2, 3, 4, 6],
                     "max_features" : [0.2, 0.4, 0.6, 0.8, 0.95],},
                    cv=cv2, 
                    refit=False, 
                    #scoring="neg_root_mean_squared_error",
                    #scoring="neg_brier_score",
                    scoring=SCORER,
                    n_jobs=4,
                    verbose=0,
                    )

dummy_grid            = deepcopy(grid)
dummy_grid.estimator  = DummyClassifier(strategy='prior')
dummy_grid.estimator  = DummyRegressor(strategy='mean')
dummy_grid.param_grid = {}

print( grid.fit(df[X], df[Y]).best_score_ )
print( grid.fit(df[selected_x], df[Y]).best_score_ )

cv_scores = []
for i in range(len(selected_x) + 1):
    grid_to_use = dummy_grid if i == 0 else grid
    s           = grid_to_use.fit(df[selected_x[:i]], df[Y]).best_score_
    cv_scores.append(s)
    print("{:<3d}: {:.6f}".format(i, cv_scores[-1]))
    
plt.plot(cv_scores, '-o', color='black')
plt.show()
#print( np.mean(cross_val_score(final, df[X], df[Y], cv = cv2, n_jobs=1, scoring=SCORER)) )
#print( np.mean(cross_val_score(final, df[selected_x], df[Y], cv = cv2, n_jobs=1, scoring=SCORER)) )