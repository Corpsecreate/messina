import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_is_fitted

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
        n_features_to_select = "auto",
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
        
        if self.n_features_to_select in (None, "auto"):
            n_to_select = n_features
        else:
            n_to_select   = min(self.n_features_to_select, n_features)
            
        n_yes         = int(self.n_permutations / 2 + 0.5)
        n_no          = self.n_permutations - n_yes
        support_      = np.zeros(n_features, dtype=bool)
        features_out_ = []
        importances_  = []
        scores        = []
        method        = 'tval'
        
        self.scoring = "neg_root_mean_squared_error"
        
        try:    
            X_NAMES = list(X.columns)
            x_type = "series"
        except: 
            X_NAMES = list(range(n_features))
            x_type = "numpy"

        SCORE_FIELD  = "__CV_SCORE__"
        peak_sample  = cross_val_score(DummyRegressor(strategy='mean'), X, y, cv = self.cv, n_jobs=-1, scoring=self.scoring)
        peak_sample  = pd.DataFrame([np.mean(peak_sample)], columns=[SCORE_FIELD])[SCORE_FIELD]
        scores.append(peak_sample)
        
        for x_i in range(n_to_select):
            
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
                sub_x                  = [x for i, x in enumerate(X_NAMES) if masks[i]]
                if x_type == "series":
                    score                  = cross_val_score(self.estimator, X[sub_x], y, cv = self.cv, n_jobs=1, scoring=self.scoring)
                else:
                    score                  = cross_val_score(self.estimator, X[:, sub_x], y, cv = self.cv, n_jobs=1, scoring=self.scoring)
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