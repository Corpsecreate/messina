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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Criterion
from sklearn.feature_selection import SequentialFeatureSelector

class ForcedSplitter(BestSplitter):
    
    def __init__(self, *args, **kwargs):
        super(BestSplitter, self).__init__(*args, **kwargs)
        
#rf      = RandomForestRegressor(n_estimators=8, max_features=0.666, max_samples=0.80, n_jobs=-1, min_impurity_decrease=min_var_split)
#t = rf.estimator
#t.splitter = ForcedSplitter(rf.criterion, rf.max_features,
#                rf.min_samples_leaf,
#                rf.min_weight_fraction_leaf,
#                rf.random_state)


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
cv            = KFold(4, shuffle=True)

min_var_split = 1e-6 * np.var(df[Y])
#base_score    = np.mean(cross_val_score(DummyClassifier(), df[X], df[Y], cv = cv, n_jobs=1, scoring=SCORER))
base_score    = np.mean(df[Y])

base_model = RandomForestRegressor(n_estimators=4, max_features=0.50, max_samples=0.20, n_jobs=-1, min_impurity_decrease=min_var_split)
base_model = RandomForestRegressor(n_estimators=16, max_features=0.80, max_samples=0.80, n_jobs=-1, min_impurity_decrease=min_var_split)
#base_model = ExtraTreesRegressor(n_estimators=512, max_features=None, n_jobs=-1, min_impurity_decrease=min_var_split)
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
N_RANDOM     = 32
n_yes        = int(N_RANDOM / 2 + 0.5)
n_no         = N_RANDOM - n_yes
SELECTED     = np.zeros(len(X))
selected_x   = []
importances_ = []
scores       = []
method       = 'tval'
alpha        = 0.20
peak_sample  = cross_val_score(DummyRegressor(strategy='mean'), df[X], df[Y], cv = cv, n_jobs=1, scoring=SCORER)
peak_sample  = pd.DataFrame([np.mean(peak_sample)], columns=['score'])['score']
scores.append(peak_sample)

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
        CONCAT_X           = sub_x# + selected_x * 4
        score              = cross_val_score(base_model, df[CONCAT_X], df[Y], cv = cv, n_jobs=1, scoring=SCORER)
        random_df['score'] = np.mean(score)
        perm_results       = pd.concat([perm_results, random_df], ignore_index=True)
    
    print(perm_results['score'].mean())
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
    if coefs[max_i] < 0:# or pvals[max_i] > alpha:
        break
    
    if peak_sample is not None:
        if n_selected == 0:
            pval = stats.ttest_1samp(perm_results['score'], peak_sample, alternative='greater').pvalue
        else:
            pval = stats.ttest_ind(perm_results['score'], peak_sample, equal_var=False, alternative='greater').pvalue
        print("Score Improvement p < {:.3f}".format(pval))
        if pval > 0.90:
            break

    
    SELECTED[X.index(max_feat)] = 1
    selected_x.append(max_feat)
    importances_.append(perm_results['score'].mean() - scores[-1].mean())
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
final = RandomForestRegressor(n_estimators=32, max_features=0.80, bootstrap=False, n_jobs=8, min_impurity_decrease=min_var_split)

grid = GridSearchCV(final,
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.tree._splitter import BestSplitter
from sklearn.tree._criterion import Criterion
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer_names

class PermutationFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that performs Sequential Feature Selection.

    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator. In
    the case of unsupervised learning, this Sequential Feature Selector
    looks only at the features (X), not the desired outputs (y).

    Read more in the :ref:`User Guide <sequential_feature_selection>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    n_features_to_select : "auto", int or float, default='warn'
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected until the score
          improvement does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

        .. versionadded:: 1.1
           The option `"auto"` was added in version 1.1.

        .. deprecated:: 1.1
           The default changed from `None` to `"warn"` in 1.1 and will become
           `"auto"` in 1.3. `None` and `'warn'` will be removed in 1.3.
           To keep the same behaviour as `None`, set
           `n_features_to_select="auto" and `tol=None`.

    tol : float, default=None
        If the score is not incremented by at least `tol` between two
        consecutive feature additions or removals, stop adding or removing.
        `tol` is enabled only when `n_features_to_select` is `"auto"`.

        .. versionadded:: 1.1

    direction : {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.

    scoring : str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        NOTE that when using a custom scorer, it should return a single
        value.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to
        add or remove, the cross-validation procedure is parallel over the
        folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_to_select_ : int
        The number of features that were selected.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    See Also
    --------
    GenericUnivariateSelect : Univariate feature selector with configurable
        strategy.
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination based on importance weights, with
        automatic selection of the number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.

    Examples
    --------
    >>> from sklearn.feature_selection import SequentialFeatureSelector
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    >>> sfs.fit(X, y)
    SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                              n_features_to_select=3)
    >>> sfs.get_support()
    array([ True, False,  True,  True])
    >>> sfs.transform(X).shape
    (150, 3)
    """

    """_parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_features_to_select": [
            StrOptions({"auto", "warn"}, deprecated={"warn"}),
            Interval(Real, 0, 1, closed="right"),
            Interval(Integral, 0, None, closed="neither"),
            Hidden(None),
        ],
        "tol": [None, Interval(Real, 0, None, closed="neither")],
        "scoring": [None, StrOptions(set(get_scorer_names())), callable],
        "cv": ["cv_object"],
        "n_jobs": [None, Integral],
    }"""

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select = "warn",
        n_permutations       = 64,
        alpha                = None,
        scoring              = None,
        cv                   = 5,
        n_jobs               = None,
    ):

        self.estimator            = estimator
        self.n_features_to_select = n_features_to_select
        self.n_permutations       = n_permutations
        self.alpha                = alpha
        self.scoring              = scoring
        self.cv                   = cv
        self.n_jobs               = n_jobs

    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        #self._validate_params()

        n_features    = X.shape[1]
        n_yes         = int(self.n_permutations / 2 + 0.5)
        n_no          = self.n_permutations - n_yes
        support_      = np.zeros(n_features, dtype=bool)
        features_out_ = []
        importances_  = []
        scores        = []
        method        = 'tval'
        
        self.scoring = "neg_root_mean_squared_error"
        
        try:    X_NAMES = list(X.columns)
        except: X_NAMES = list(range(n_features))

        SCORE_FIELD  = "__CV_SCORE__"
        peak_sample  = cross_val_score(DummyRegressor(strategy='mean'), X, y, cv = self.cv, n_jobs=-1, scoring=self.scoring)
        peak_sample  = pd.DataFrame([np.mean(peak_sample)], columns=[SCORE_FIELD])[SCORE_FIELD]
        scores.append(peak_sample)
        
        for x_i in range(min(self.n_features_to_select, n_features)):
            
            flags        = np.repeat([True] * n_yes + [False] * n_no, n_features).reshape(self.n_permutations, -1)
            rng          = np.random.default_rng()
            flags        = rng.permuted(flags, axis=0)
            perm_results = pd.DataFrame()
            
            for perm_i in tqdm(range(self.n_permutations)):
                
                if perm_i == 0:
                    masks = [i % 2 == 0 for i in range(n_features)]
                elif perm_i == 1:
                    masks = [i % 2 == 1 for i in range(n_features)]
                else:
                    masks = flags[perm_i]
                    while sum(masks) == 0:
                        masks = np.random.choice([True, False], n_features)
                    
                masks                  = [True if support_[i] == True else x for i, x in enumerate(masks)]
                random_df              = pd.DataFrame(np.array(masks, dtype=bool).reshape(-1, len(masks)), columns=X_NAMES)
                sub_x                  = [x for i, x in enumerate(X) if masks[i]]
                score                  = cross_val_score(self.estimator, X[sub_x], y, cv = self.cv, n_jobs=1, scoring=self.scoring)
                random_df[SCORE_FIELD] = np.mean(score)
                perm_results           = pd.concat([perm_results, random_df], ignore_index=True)
            
            #print(perm_results[SCORE_FIELD].mean())
            remain_x = [x for i, x in enumerate(X_NAMES) if support_[i] == False]
            coefs    = [0] * len(remain_x)
            pvals    = [0] * len(remain_x)
            tstats   = [0] * len(remain_x)
            
            for i, x in enumerate(remain_x):
                when_on     = perm_results[perm_results[x] == True]
                when_off    = perm_results[perm_results[x] == False]
                test_result = stats.ttest_ind(when_on[SCORE_FIELD], when_off[SCORE_FIELD], equal_var=False, alternative='greater')
                coefs[i]    = when_on[SCORE_FIELD].mean() - when_off[SCORE_FIELD].mean()
                pvals[i]    = test_result.pvalue
                tstats[i]   = test_result.statistic
                
            max_i    = np.argmax(coefs if method == 'coef' else tstats)
            max_feat = remain_x[max_i]
            
            print("{:<16} {:<12,.6f} {:<12.6f}".format(max_feat, coefs[max_i], pvals[max_i]))
            if coefs[max_i] < 0:
                break
            
            if x_i == 0:
                pval = stats.ttest_1samp(perm_results[SCORE_FIELD], peak_sample, alternative='greater').pvalue
            else:
                pval = stats.ttest_ind(perm_results[SCORE_FIELD], peak_sample, equal_var=False, alternative='greater').pvalue
            print("Score Improvement p < {:.3f}".format(pval))
            if self.alpha is not None and pval > self.alpha:
                break

            support_[X_NAMES.index(max_feat)] = True
            features_out_.append(max_feat)
            importances_.append(coefs[max_i])
            
            scores.append(perm_results[SCORE_FIELD])
            if perm_results[SCORE_FIELD].mean() >= peak_sample.mean():
                peak_sample = perm_results[SCORE_FIELD]
                
        self.support_      = support_
        self.features_out_ = features_out_
        self.importances_  = importances_
        self.cv_scores_    = [s.mean() for s in scores]

        return self
    
    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_
    
pf = PermutationFeatureSelector(DecisionTreeRegressor(), n_features_to_select=10, n_permutations=64)
pf.fit(df[X], df[Y])

importances_norm_ = [x / sum(pf.importances_) for x in pf.importances_]

fig, ax = plt.subplots()
ax.barh(pf.features_out_[::-1], importances_norm_[::-1], color='dodgerblue')
#ax.set_label([f'{p:.1f}%' for p in importances_norm_[::-1]])
ax.bar_label(ax.containers[0], labels=[f'{100*p:.1f}%' for p in importances_norm_[::-1]])
plt.show()

plt.plot([s.mean() for s in pf.cv_scores_], '-o')
plt.show()
