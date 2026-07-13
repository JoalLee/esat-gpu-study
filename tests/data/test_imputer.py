import pytest
import numpy as np
import pandas as pd

from esat.data.impute import DataImputer
from esat.data.datahandler import DataHandler

@pytest.fixture
def df_with_nans():
    # Larger DataFrame: 6 rows, 4 columns + location
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, np.nan],
        'B': [np.nan, 2, 3, 4, np.nan, 6],
        'C': ['x', 'y', 'z', None, 'x', 'y'],
        'D': [10, np.nan, 12, 13, 14, 15],
        'location': ['loc1', 'loc1', 'loc2', 'loc2', 'loc3', 'loc3']
    })

@pytest.fixture
def df_uncertainty(df_with_nans):
    # Same shape, small float values for uncertainty
    return pd.DataFrame({
        'A': [0.01, 0.02, 0.01, 0.03, 0.01, 0.02],
        'B': [0.02, 0.01, 0.01, 0.02, 0.01, 0.03],
        'C': [0, 0, 0, 0, 0, 0],  # categorical, so 0
        'D': [0.01, 0.01, 0.02, 0.01, 0.01, 0.02],
        'location': ['loc1', 'loc1', 'loc2', 'loc2', 'loc3', 'loc3']
    })

@pytest.fixture
def handler_with_nans(df_with_nans, df_uncertainty):
    return DataHandler.load_dataframe(df_with_nans, df_uncertainty)

@pytest.mark.parametrize("strategy", ['mean', 'median', 'most_frequent', 'knn', 'iterative'])
def test_impute_supported_strategies(handler_with_nans, df_with_nans, strategy):
    imputer = DataImputer(handler_with_nans)
    if strategy == 'most_frequent':
        imputer.impute(strategy=strategy)
        assert imputer.imputed_data['C'].iloc[0] == 'x'
        assert imputer.imputed_data['C'].iloc[3] == 'x'
    else:
        imputer.impute(strategy=strategy)
        assert not imputer.imputed_data[['A', 'B', 'D']].isnull().any().any()
    assert imputer.imputed_data.columns.equals(df_with_nans.columns)
    assert imputer.imputed_uncertainty.columns.equals(df_with_nans.columns)

def test_impute_invalid_strategy(handler_with_nans):
    imputer = DataImputer(handler_with_nans)
    with pytest.raises(ValueError):
        imputer.impute(strategy='unsupported')

def test_impute_extra_column(df_with_nans, df_uncertainty):
    df_extra = df_with_nans.copy()
    df_extra['E'] = [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="identical columns"):
        DataHandler.load_dataframe(df_extra, df_uncertainty)

def test_impute_missing_column(df_with_nans, df_uncertainty):
    df_missing = df_with_nans.drop(columns=['A'])
    with pytest.raises(ValueError, match="identical columns"):
        DataHandler.load_dataframe(df_missing, df_uncertainty)

def test_impute_without_handler():
    with pytest.raises(TypeError):
        DataImputer().impute(strategy='mean')
