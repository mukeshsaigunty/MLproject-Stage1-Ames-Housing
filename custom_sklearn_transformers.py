from imports_and_env_setup import *

class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.01):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        self.maps_ = {}
        n = len(X)
        for col in X.columns:
            freq = X[col].value_counts(dropna=False) / n
            self.maps_[col] = set(freq[freq < self.min_freq].index)
        return self

    def transform(self, X):
        X = X.copy()
        for col, rares in self.maps_.items():
            X[col] = X[col].where(~X[col].isin(rares), "__RARE__")
        return X
