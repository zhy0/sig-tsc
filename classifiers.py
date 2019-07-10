import numpy as np
from iisignature import sig, logsig, prepare

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


class SigFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, level=3):
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sig(x, self.level) for x in X])


class LogSigFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, level=3, dim=2):
        self.level = level
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prepared = prepare(self.dim, self.level)
        return np.array([logsig(x, prepared) for x in X])


class Embedding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        raise NotImplementedError

    def transform(self, X):
        return [self.transform_instance(x) for x in X]


class LeadLag(Embedding):
    def transform_instance(self, X):
        lead = np.transpose([X, X]).flatten()[1:]
        lag =  np.transpose([X, X]).flatten()[0:-1]
        return np.transpose([lead, lag])


class TimeIndexed(Embedding):
    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]


class TimeJoined(Embedding):
    def transform_instance(self, X):
        Y = X.transpose()
        t = np.transpose([Y[0], Y[0]]).flatten()
        Z = np.insert(np.transpose([Y[1], Y[1]]).flatten()[0:-1], 0,0)
        return np.transpose([t,Z])


class FlatCOTE(VotingClassifier):
    def __init__(self, estimators, cv=3, n_jobs=None, flatten_transform=True):
        super().__init__(estimators, voting='soft', weights=None, n_jobs=n_jobs,
                         flatten_transform=flatten_transform)
        self.cv = cv

    def fit(self, X, y):
        super().fit(X, y)
        self.weights = [cross_val_score(clf, X, y, cv=self.cv).mean()
                        for clf in self.estimators_]
        return self


def create_concatenator(clf, sig_type='logsig', level=3, dim=2):
    if sig_type == 'logsig':
        if not dim:
            raise
        sig_features = LogSigFeatures(level=level, dim=dim)
    else:
        sig_features = SigFeatures(level=level)

    leadlag = Pipeline([
        ('leadlag',    LeadLag()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    timeindexed = Pipeline([
        ('timeind',    TimeIndexed()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    timejoined = Pipeline([
        ('timeind',    TimeIndexed()),
        ('timejoin',   TimeJoined()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    partial_sum = lambda X : np.cumsum(X, axis=1)
    ps_leadlag = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('leadlag',    LeadLag()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    ps_timeindexed = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('timeind',    TimeIndexed()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    ps_timejoined = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('timeind',    TimeIndexed()),
        ('timejoin',   TimeJoined()),
        ('signature',  sig_features),
        ('scale',      StandardScaler()),
    ])

    union = FeatureUnion([
        ('leadlag', leadlag),
        ('timejoined', timejoined),
        ('timeindexed', timeindexed),
        ('ps_leadlag', ps_leadlag),
        ('ps_timejoined', ps_timejoined),
        ('ps_timeindexed', ps_timeindexed),
    ])

    return Pipeline([
        ('union', union),
        ('classifier', clf)
    ])

def create_vote_clf(clf, level=3, voter=FlatCOTE, **vote_args):
    leadlag = Pipeline([
        ('leadlag',    LeadLag()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    timeindexed = Pipeline([
        ('timeind',    TimeIndexed()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    timejoined = Pipeline([
        ('timeind',    TimeIndexed()),
        ('timejoin',   TimeJoined()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    partial_sum = lambda X : np.cumsum(X, axis=1)
    ps_leadlag = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('leadlag',    LeadLag()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    ps_timeindexed = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('timeind',    TimeIndexed()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    ps_timejoined = Pipeline([
        ('partialsum', FunctionTransformer(partial_sum, validate=False)),
        ('timeind',    TimeIndexed()),
        ('timejoin',   TimeJoined()),
        ('signature',  SigFeatures(level=level)),
        ('scale',      StandardScaler()),
        ('classifier', clone(clf)),
    ])

    vote = voter([
        ('leadlag', leadlag),
        ('timejoined', timejoined),
        ('timeindexed', timeindexed),
        ('ps_leadlag', ps_leadlag),
        ('ps_timejoined', ps_timejoined),
        ('ps_timeindexed', ps_timeindexed),
    ], **vote_args)

    return vote
