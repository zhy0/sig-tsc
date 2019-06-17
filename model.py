#!/usr/bin/env python3

import numpy as np
from iisignature import sig, logsig, prepare
from sklearn import preprocessing


class SigModel:
    """
    Signature classification/regression model.

    Params:
        model: (class) sklearn classification/regression model to use
        level: (int) signature truncation level to use
        transform: (callable) path embedding function, e.g., SigModel.leadlag
        scale: (bool) whether to apply column-wise scaling of signature features
            using to sklearn.preprocessing.scale.
    """

    def __init__(self, model, level=2, transform=lambda x: x, scale=True, **model_args):
        self.level = level
        self.model = model(**model_args)
        self.transform = transform
        self.scale = scale

    def preprocess(self, X):
        """
        Preprocess training/testing data using signatures.
        """
        data = [sig(self.transform(x), self.level) for x in X]
        if self.scale:
            data = preprocessing.scale(data)
        return data

    def train(self, X, Y):
        """Train the signature model"""
        assert len(X) == len(Y)
        self.model.fit(self.preprocess(X), Y)

    def predict(self, X):
        """Predict using trained model"""
        return self.model.predict(self.preprocess(X))

    def score(self, X, Y):
        """Output score of trained model, depends on used model"""
        return self.model.score(self.preprocess(X), Y)


    @staticmethod
    def time_indexed(X):
        """
        Turn 1-dimensional list into 2-dimensional list of points by adding
        list index.

        Params:
            X: (list) 1-dimensional list of length N to be transformed

        Returns: (list) 2-dimensional list of shape (N, 2)
        """
        if not np.shape(X) == (len(X),):
            raise ValueError("Input does not have correct shape")

        return np.transpose([np.arange(len(X)), X])


    @staticmethod
    def lead_lag(X):
        """
        Compute lead-lag transformation of 1-dimensional list of values.

        Params:
            X: (list) 1-dimensional list of length N to be transformed

        Returns: (list) 2-dimensional list of shape (N, 2)
        """
        if not np.shape(X) == (len(X),):
            raise ValueError("Input does not have correct shape")

        lead = np.transpose([X, X]).flatten()[1:]
        lag =  np.transpose([X, X]).flatten()[0:-1]
        return np.transpose([lead, lag])


    @staticmethod
    def time_joined(X):
        """
        Compute time-joined transformation of a path.

        Params:
            X: (list) a list of shape (N,2) or (N,) with N length of path; in
                the case of (N,2), the first component of the path must be the
                time index.

        Returns: (list) dimensional list of shape (N, 2)
        """
        if np.shape(X) == (len(X),):
            # if there is no time index, we simply use the list index
            Y = np.array([np.arange(len(X)), X])
        elif np.shape(X) == (len(X), 2):
            Y = np.transpose(X)
        else:
            raise ValueError("Input does not have correct shape")

        t = np.transpose([Y[0], Y[0]]).flatten()
        Z = np.insert(np.transpose([Y[1], Y[1]]).flatten()[0:-1], 0,0)
        return np.transpose([t,Z])


class LogSigModel(SigModel):
    """
    Classification/regression model using log signature features.

    Params:
        model: (class) sklearn classification/regression model to use
        dim: dimension of transformed path, needed for iisignature
        level: (int) signature truncation level to use
        transform: (callable) path embedding function, e.g., SigModel.leadlag
        scale: (bool) whether to apply column-wise scaling of signature features
            using to sklearn.preprocessing.scale.
    """

    def __init__(self, model, dim, level=2, transform=lambda x: x, **model_args):
        self.prepared = prepare(dim, level) # iisignature prepare log signature
        super().__init__(model, level, transform, **model_args)

    def preprocess(self, X):
        return [logsig(self.transform(x), self.prepared) for x in X]
