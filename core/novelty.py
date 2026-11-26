"""Novelty-mode (inductive) scoring for PyOD's COF, SOD, ECOD and COPOD.

PyOD's COF and SOD are transductive: ``decision_function(X)`` builds every
neighborhood inside ``X`` itself and ignores the data the detector was fitted
on. Scoring a test split or a batch of synthetic points that way measures
them against each other, not against the training data. A sparse tail point
then looks normal as soon as a few other new points sit near it.

ECOD and COPOD add the scored batch to the training data before they compute
their empirical CDFs, so values beyond the training data are ranked against
each other: the same value scores differently depending on its batch.

The subclasses below keep PyOD's training scores (``decision_scores_``, each
training point scored within the training set, itself left out) and score
new points against the stored training set only. A new point counts as added
to the training set on its own, without changing the training points' own
statistics, which is the usual novelty-detection approximation (as in
scikit-learn's ``LocalOutlierFactor(novelty=True)``).
"""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.sod import SOD

# Rows scored per vectorized block (bounds the temporary arrays).
_BLOCK_ROWS = 512


def _blocks(n_rows):
    for start in range(0, n_rows, _BLOCK_ROWS):
        yield slice(start, min(start + _BLOCK_ROWS, n_rows))


def _average_chaining_distance(paths):
    """COF's average chaining distance for each SBN path, exactly as PyOD
    computes it: ``paths`` is ``(m, k + 1, d)``, the point itself followed by
    its k nearest neighbors in order. The j-th cost is the shortest distance
    from neighbor j to the point or an earlier neighbor, weighted by
    ``2 (k + 1 - j) / ((k + 1) k)``.
    """
    k = paths.shape[1] - 1
    distances = np.linalg.norm(paths[:, :, None, :] - paths[:, None, :, :], axis=-1)
    earlier = np.tril(np.ones((k + 1, k + 1), dtype=bool), -1)
    costs = np.where(earlier, distances, np.inf)[:, 1:, :].min(axis=2)
    j = np.arange(1, k + 1)
    weights = 2.0 * (k + 1 - j) / ((k + 1) * k)
    return costs @ weights


class InductiveCOF(COF):
    """COF that scores new points against its training set (see module
    docstring). Training scores are PyOD's own, so ``method`` still applies
    to them.
    """

    def fit(self, X, y=None):
        for name in ("_x_fit", "_neighbors", "_train_acd"):
            self.__dict__.pop(name, None)
        super().fit(X, y)  # transductive scores on the training set
        X = check_array(X)
        k = self.n_neighbors_
        self._neighbors = NearestNeighbors(n_neighbors=k).fit(X)
        own = self._neighbors.kneighbors(return_distance=False)  # self left out
        acd = np.empty(len(X))
        for rows in _blocks(len(X)):
            paths = np.concatenate([X[rows, None, :], X[own[rows]]], axis=1)
            acd[rows] = _average_chaining_distance(paths)
        self._train_acd = acd
        self._x_fit = X
        return self

    def decision_function(self, X):
        if not hasattr(self, "_x_fit"):  # still inside fit(): PyOD's scores
            return super().decision_function(X)
        X = check_array(X)
        k = self.n_neighbors_
        scores = np.empty(len(X))
        for rows in _blocks(len(X)):
            neighbors = self._neighbors.kneighbors(X[rows], return_distance=False)
            paths = np.concatenate([X[rows, None, :], self._x_fit[neighbors]], axis=1)
            acd = _average_chaining_distance(paths)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores[rows] = acd * k / self._train_acd[neighbors].sum(axis=1)
        return np.nan_to_num(scores)


class InductiveSOD(SOD):
    """SOD that scores new points against its training set (see module
    docstring): a new point's reference set is the ``ref_set`` training
    points sharing the most of its k nearest training neighbors.
    """

    def fit(self, X, y=None):
        for name in ("_x_fit", "_neighbors", "_neighbor_sets"):
            self.__dict__.pop(name, None)
        super().fit(X, y)  # transductive scores on the training set
        X = check_array(X)
        self._neighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric, p=self.p,
            metric_params=self.metric_params, n_jobs=self.n_jobs,
        ).fit(X)
        own = self._neighbors.kneighbors(return_distance=False)  # self left out
        self._neighbor_sets = self._indicator(own, len(X))
        self._x_fit = X
        return self

    @staticmethod
    def _indicator(neighbor_indices, n_columns):
        """Sparse 0/1 matrix: row i has ones at row i's neighbor indices."""
        n_rows, k = neighbor_indices.shape
        return sparse.csr_matrix(
            (np.ones(n_rows * k), neighbor_indices.ravel(),
             np.arange(0, n_rows * k + 1, k)),
            shape=(n_rows, n_columns),
        )

    def decision_function(self, X):
        if not hasattr(self, "_x_fit"):  # still inside fit(): PyOD's scores
            return super().decision_function(X)
        X = check_array(X)
        n_train, n_dims = self._x_fit.shape
        scores = np.zeros(len(X))
        for rows in _blocks(len(X)):
            neighbors = self._neighbors.kneighbors(X[rows], return_distance=False)
            shared = (self._indicator(neighbors, n_train) @ self._neighbor_sets.T).toarray()
            reference = np.argsort(-shared, axis=1, kind="stable")[:, :self.ref_set]
            ref = self._x_fit[reference]  # (m, ref_set, d)
            means = ref.mean(axis=1)
            var_total = np.square(ref - means[:, None, :]).sum(axis=(1, 2)) / self.ref_set
            var_expect = self.alpha * var_total / n_dims
            relevant = ref.var(axis=1) < var_expect[:, None]
            rel_dim = relevant.sum(axis=1)
            deviation = (relevant * np.square(X[rows] - means)).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores[rows] = np.where(rel_dim > 0, np.sqrt(deviation / rel_dim), 0.0)
        return scores


class _InductiveTails:
    """Scores every new row exactly as PyOD scores it alone, i.e. as the only
    row added to the training data, but for all rows at once.
    """

    def fit(self, X, y=None):
        # PyOD's decision_function adds X_train to what it scores, which must
        # not happen while a second fit computes the training scores.
        for name in ("X_train", "_tails"):
            self.__dict__.pop(name, None)
        super().fit(X, y)
        X = self.X_train
        mean = X.mean(axis=0)
        centered = X - mean
        self._tails = (np.sort(X, axis=0), mean, centered.sum(axis=0),
                       (centered ** 2).sum(axis=0), (centered ** 3).sum(axis=0))
        return self

    def decision_function(self, X):
        if not hasattr(self, "_tails"):  # still inside fit(): PyOD's scores
            return super().decision_function(X)
        X = check_array(X)
        ordered, mean, s1, s2, s3 = self._tails
        n = len(ordered) + 1  # the training rows and the new one
        at_most = np.empty(X.shape)
        at_least = np.empty(X.shape)
        for c in range(X.shape[1]):
            at_most[:, c] = np.searchsorted(ordered[:, c], X[:, c], side="right") + 1
            at_least[:, c] = n - np.searchsorted(ordered[:, c], X[:, c], side="left")
        left, right = -np.log(at_most / n), -np.log(at_least / n)

        # Skewness of each column with the new value included.
        d = X - mean
        shift = (s1 + d) / n
        m2 = (s2 + d ** 2) / n - shift ** 2
        m3 = (s3 + d ** 3) / n - 3 * shift * (s2 + d ** 2) / n + 2 * shift ** 3
        sign = np.where(m2 > 0, np.sign(m3), 0.0)
        skewed = np.where(sign > 0, right, np.where(sign < 0, left, left + right))
        return self._combine(left, right, skewed).sum(axis=1)


class InductiveECOD(_InductiveTails, ECOD):
    @staticmethod
    def _combine(left, right, skewed):
        return np.maximum(skewed, np.maximum(left, right))


class InductiveCOPOD(_InductiveTails, COPOD):
    @staticmethod
    def _combine(left, right, skewed):
        return np.maximum(skewed, (left + right) / 2)
