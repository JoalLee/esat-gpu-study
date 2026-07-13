import logging

import pandas as pd
import numpy as np

from esat.data.datahandler import DataHandler

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    sklearn_available = True
except ImportError:
    sklearn_available = False

logger = logging.getLogger(__name__)


class DataImputer:
    """
    Class for imputing missing values in datasets using various strategies.

    This class supports mean, median, most_frequent, KNN, and iterative imputation strategies.

    :param data_handler: An instance of DataHandler containing the dataset to be imputed.
    :param random_seed: Random seed for reproducibility (default is 42).
    :param missing_value: Value to be treated as missing (default is np.nan).
    """
    def __init__(self, data_handler: DataHandler, random_seed: int = 42, missing_value: float = np.nan):
        self.data_handler = data_handler
        self.random_seed = random_seed

        self.imputation_mask = None
        self.imputed_data = None
        self.imputed_uncertainty = None

        self.strategy = None
        self.missing_value = missing_value

        if not sklearn_available:
            raise ImportError("scikit-learn is required for data imputation. Import esat[data] to install it.")

    def impute(self, strategy='mean', args: dict = None):
        """
        Impute missing values in the dataset using the specified strategy.

        :param strategy: The imputation strategy to use ('mean', 'median', 'most_frequent', 'knn', or 'iterative').
        :param args: Additional arguments for the imputer (e.g., n_neighbors for KNN, max_iter for IterativeImputer).
        """
        if strategy not in ['mean', 'median', 'most_frequent', 'knn', 'iterative']:
            raise ValueError(f"Invalid imputation strategy: {strategy}. Choose from 'mean', 'median', 'most_frequent', "
                             f"'knn', or 'iterative'.")
        if self.data_handler is None:
            logger.warning("No data to impute. The dataset is empty.")
            return
        self.strategy = strategy
        logger.info(f"Imputing missing values using strategy: {strategy}")
        V = self.data_handler.input_data.copy()
        U = self.data_handler.uncertainty_data.copy()
        if not V.columns.equals(U.columns):
            raise ValueError("Input and uncertainty data must have identical columns in the same order")
        if not V.index.equals(U.index):
            raise ValueError("Input and uncertainty data must use the same row index")
        if self.missing_value is not None:
            V = V.mask(V == self.missing_value)
            U = U.mask(U == self.missing_value)
        V = V.mask(V.isna())
        U = U.mask(U.isna())

        # Create a mask for referencing missing values
        self.imputation_mask = V.isna() | U.isna()

        numeric_columns = V.select_dtypes(include=[np.number]).columns.intersection(
            U.select_dtypes(include=[np.number]).columns,
            sort=False,
        )
        other_columns = V.columns.difference(numeric_columns, sort=False)

        self.imputed_data = V.copy()
        self.imputed_uncertainty = U.copy()
        if len(other_columns):
            self._impute_columns(
                V,
                U,
                other_columns,
                SimpleImputer(strategy='most_frequent', keep_empty_features=True),
            )

        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy, missing_values=np.nan, keep_empty_features=True)
        elif strategy == 'knn':
            if args is None:
                args = {}
            n_neighbors = args.get('n_neighbors', 5)
            weights = args.get('weights', 'uniform')
            imputer = KNNImputer(
                n_neighbors=n_neighbors,
                weights=weights,
                missing_values=np.nan,
                keep_empty_features=True,
            )
        elif strategy == 'iterative':
            if args is None:
                args = {}
            max_iter = args.get('max_iter', 10)
            tol = args.get('tol', 1e-3)
            imputer = IterativeImputer(
                max_iter=max_iter,
                tol=tol,
                random_state=self.random_seed,
                missing_values=np.nan,
                keep_empty_features=True,
            )
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}. Supported strategies are 'mean', 'median', "
                             f"'most_frequent', 'knn', and 'iterative'.")

        if len(numeric_columns):
            self._impute_columns(V, U, numeric_columns, imputer)
        return self.imputed_data, self.imputed_uncertainty

    def _impute_columns(self, V, U, columns, imputer):
        """Impute matching columns while retaining the original index and column order."""
        data_values = imputer.fit_transform(V.loc[:, columns])
        uncertainty_values = imputer.__class__(**imputer.get_params()).fit_transform(U.loc[:, columns])
        self.imputed_data.loc[:, columns] = data_values
        self.imputed_uncertainty.loc[:, columns] = uncertainty_values

    def _run_simple_imputer(self, V, U, strategy='mean'):
        """
        Run the SimpleImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param strategy: The imputation strategy to use ('mean', 'median', 'most_frequent').
        :return: Imputed data.
        """
        imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)
        imputer2 = SimpleImputer(strategy=strategy, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=U.columns, index=U.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=U.columns, index=U.index)
        return self.imputed_data, self.imputed_uncertainty

    def _run_knn_imputer(self, V, U, n_neighbors=5, weights='uniform'):
        """
        Run the KNNImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param n_neighbors: Number of neighbors to use for imputation.
        :param weights: Weight function used in prediction ('uniform' or 'distance').
        :return: Imputed data.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=np.nan)
        imputer2 = KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=U.columns, index=U.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=U.columns, index=U.index)
        return self.imputed_data, self.imputed_uncertainty

    def _run_iterative_imputer(self, V, U, max_iter=10, tol=1e-3):
        """
        Run the IterativeImputer from scikit-learn on the provided data.

        :param U: The input data to impute.
        :param V: The uncertainty data (not used in this method).
        :param max_iter: Maximum number of imputation iterations.
        :param tol: Tolerance for stopping criteria.
        :return: Imputed data.
        """
        imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=self.random_seed, missing_values=np.nan)
        imputer2 = IterativeImputer(max_iter=max_iter, tol=tol, random_state=self.random_seed, missing_values=np.nan)
        imputed_data = imputer.fit_transform(V)
        imputed_uncertainty = imputer2.fit_transform(U)
        self.imputed_data = pd.DataFrame(imputed_data, columns=U.columns, index=U.index)
        self.imputed_uncertainty = pd.DataFrame(imputed_uncertainty, columns=U.columns, index=U.index)
        return self.imputed_data, self.imputed_uncertainty
