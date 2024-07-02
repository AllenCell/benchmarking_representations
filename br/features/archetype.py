"""Implementation of Frank-Wolfe algorithm for the archetypal analysis.

Algorithm is based on the paper: "Archetypal Analysis as an Autoencoder"
(https://www.researchgate.net/publication/282733207_Archetypal_Analysis_as_an_Autoencoder)
Code adapted from https://github.com/atmguille/archetypal-analysis/blob/main/Python%20implementation/AA_Fast.py
"""

from abc import ABC, abstractmethod

import numpy as np


class AA_Abstract(ABC):
    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.Z = None  # Archetypes
        self.n_samples, self.n_features = None, None
        self.RSS = None

    def fit(self, X: np.ndarray) -> "AA_Abstract":
        """Computes the archetypes and the RSS from the data X, which are stored in the
        corresponding attributes.

        :param X: data matrix, with shape (n_samples, n_features)
        :return: self
        """
        self.n_samples, self.n_features = X.shape
        self._fit(X)
        return self

    def _fit(self, X: np.ndarray):
        """Internal function that computes the archetypes and the RSS from the data X.

        :param X: data matrix, with shape (n_samples, n_features)
        :return: None
        """
        # Initialize the archetypes
        B = np.eye(self.n_archetypes, self.n_samples)
        Z = B @ X

        A = np.eye(self.n_samples, self.n_archetypes)
        prev_RSS = None

        for _ in range(self.max_iter):
            A = self._computeA(X, Z, A)
            B = self._computeB(X, A, B)
            Z = B @ X
            RSS = self._rss(X, A, Z)
            if prev_RSS is not None and abs(prev_RSS - RSS) / prev_RSS < self.tol:
                break
            prev_RSS = RSS

        self.Z = Z
        self.RSS = RSS

    @staticmethod
    @abstractmethod
    def _computeA(X: np.ndarray, Z: np.ndarray, A: np.ndarray = None) -> np.ndarray:
        """Updates the A matrix given the data matrix X and the archetypes Z. A is the matrix that
        gives the best convex approximation of X by Z, so this function can be used during training
        and inference. For the latter, use the transform method instead.

        :param X: data matrix, with shape (n_samples, n_features)
        :param Z: archetypes matrix, with shape (n_archetypes, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :return: A matrix, with shape (n_samples, n_archetypes)
        """
        pass

    @staticmethod
    @abstractmethod
    def _computeB(X: np.ndarray, A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
        """Updates the B matrix given the data matrix X and the A matrix.

        :param X: data matrix, with shape (n_samples, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :param B: B matrix, with shape (n_archetypes, n_samples)
        :return: B matrix, with shape (n_archetypes, n_samples)
        """
        pass

    def archetypes(self) -> np.ndarray:
        """
        Returns the archetypes' matrix
        :return: archetypes matrix, with shape (n_archetypes, n_features)
        """
        return self.Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Computes the best convex approximation A of X by the archetypes.

        :param X: data matrix, with shape (n_samples, n_features)
        :return: A matrix, with shape (n_samples, n_archetypes)
        """
        return self._computeA(X, self.Z)

    @staticmethod
    def _rss(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> float:
        """Computes the RSS of the data matrix X, given the A matrix and the archetypes Z.

        :param X: data matrix, with shape (n_samples, n_features)
        :param A: A matrix, with shape (n_samples, n_archetypes)
        :param Z: archetypes matrix, with shape (n_archetypes, n_features)
        :return: RSS
        """
        return np.linalg.norm(X - A @ Z) ** 2


class AA_Fast(AA_Abstract):
    def __init__(
        self,
        n_archetypes: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        derivative_max_iter: int = 10,
    ):
        super().__init__(n_archetypes, max_iter, tol, verbose)
        self.derivative_max_iter = derivative_max_iter

    def _computeA(
        self, X: np.ndarray, Z: np.ndarray, A: np.ndarray = None
    ) -> np.ndarray:
        A = np.zeros((self.n_samples, self.n_archetypes))
        A[:, 0] = 1.0
        e = np.zeros(A.shape)
        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            G = 2.0 * (A @ (Z @ Z.T) - X @ Z.T)
            # Get the argument mins along each column
            argmins = np.argmin(G, axis=1)
            e[range(self.n_samples), argmins] = 1.0
            A += 2.0 / (t + 2.0) * (e - A)
            e[range(self.n_samples), argmins] = 0.0
        return A

    def _computeB(
        self, X: np.ndarray, A: np.ndarray, B: np.ndarray = None
    ) -> np.ndarray:
        B = np.zeros((self.n_archetypes, self.n_samples))
        B[:, 0] = 1.0
        e = np.zeros(B.shape)
        for t in range(self.derivative_max_iter):
            # brackets are VERY important to save time
            t1 = (A.T @ A) @ (B @ X) @ X.T
            t2 = (A.T @ X) @ X.T
            G = 2.0 * (t1 - t2)
            argmins = np.argmin(G, axis=1)
            e[range(self.n_archetypes), argmins] = 1.0
            B += 2.0 / (t + 2.0) * (e - B)
            e[range(self.n_archetypes), argmins] = 0.0
        return B
