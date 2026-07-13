import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from esat.data.datahandler import DataHandler
from esat.data.analysis import BatchAnalysis, ModelAnalysis

logger = logging.getLogger(__name__)


class TestDataHandler:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    batch_name = "bs_test00"

    @classmethod
    def setup_class(self):
        logger.info("Running SA Test Setup")
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.input_file = os.path.join(self.data_path, "Dataset-BatonRouge-con.csv")
        self.uncertainty_file = os.path.join(self.data_path, "Dataset-BatonRouge-unc.csv")

    def test_load(self):
        datahandler = DataHandler(
            input_path=self.input_file,
            uncertainty_path=self.uncertainty_file,
            index_col='Date'
        )
        V, U = datahandler.get_data()
        assert V.shape == (307, 41)

    def test_load_dataframe(self):
        input_df = pd.read_csv(self.input_file, index_col="Date")
        uncertainty_df = pd.read_csv(self.uncertainty_file, index_col="Date")
        datahandler = DataHandler.load_dataframe(input_df=input_df, uncertainty_df=uncertainty_df)
        V, U = datahandler.get_data()
        assert V.shape == (307, 41)

    def test_aggregation_with_duplicates(self):
        n_samples = 15
        input_df = pd.DataFrame({
            'normal': np.linspace(0, 100, n_samples),
            'constant': [5.0] * n_samples
        }, index=pd.date_range(start='2020-01-01', periods=n_samples, freq='D', name='Date'))
        uncertainty_df = pd.DataFrame({
            'normal': np.full(n_samples, 1.0),
            'constant': np.full(n_samples, 0.5),
        }, index=input_df.index)

        dh = DataHandler(input_path="", uncertainty_path="", load=False, max_plotting_n=10)
        dh.input_data = input_df
        dh.uncertainty_data = uncertainty_df
        dh.features = input_df.columns
        dh._load_data(existing_data=True)
        dh._determine_optimal_block()
        dh.get_data()
        dh._aggregate_data()

        assert dh.input_data_plot.shape == (10, 2)
        assert dh.uncertainty_data_plot.shape == (10, 2)
        assert dh.input_data_plot.notna().all().all()
        assert dh.uncertainty_data_plot.notna().all().all()
        assert (dh.input_data_plot['constant'] == 5.0).all()
        assert dh.input_data_plot.index.equals(dh.uncertainty_data_plot.index)
        assert dh.input_data_plot.index.is_monotonic_increasing

        output = np.column_stack((np.arange(n_samples), np.full(n_samples, 3.0)))
        aggregated_output = dh.aggregate_output(output)
        assert aggregated_output.shape == (10, 2)
        assert aggregated_output.index.equals(dh.input_data_plot.index)
        assert aggregated_output.notna().all().all()

        model = SimpleNamespace(
            W=np.ones((n_samples, 2)),
            H=np.array([[1.0, 2.0], [0.5, 1.5]]),
            factors=2,
        )
        _, factor_vprime = ModelAnalysis(datahandler=dh, model=model).aggregate_factors_for_plotting()
        assert factor_vprime.shape == (10, 2)
        assert factor_vprime.index.equals(dh.input_data_plot.index)
        assert factor_vprime.notna().all().all()

        batch = SimpleNamespace(
            V=input_df.to_numpy(),
            results=[SimpleNamespace(WH=np.ones((n_samples, 2)))],
            best_model=0,
        )
        residual_fig = BatchAnalysis(batch_sa=batch, data_handler=dh).plot_temporal_residuals(
            feature_idx=0, show=False
        )
        assert len(residual_fig.data) == 2
        assert len(residual_fig.data[0].x) == 10
        assert len(residual_fig.data[1].y) == 10

    def test_aggregation_rejects_misaligned_uncertainty_index(self):
        input_df = pd.DataFrame(
            {'value': [1.0, 2.0, 3.0]},
            index=pd.date_range('2020-01-01', periods=3, freq='D'),
        )
        uncertainty_df = pd.DataFrame(
            {'value': [0.1, 0.2, 0.3]},
            index=pd.date_range('2020-01-02', periods=3, freq='D'),
        )
        dh = DataHandler(input_path="", uncertainty_path="", load=False, max_plotting_n=2)
        dh.input_data = input_df
        dh.uncertainty_data = uncertainty_df

        with pytest.raises(ValueError, match="same row index"):
            dh._aggregate_data()

    def test_get_data_drops_location_before_numeric_conversion(self):
        input_df = pd.DataFrame({
            'value': [1.0, 2.0],
            'location': ['north', 'south'],
        })
        uncertainty_df = pd.DataFrame({
            'value': [0.1, 0.2],
            'location': ['north', 'south'],
        })
        dh = DataHandler(input_path="", uncertainty_path="", load=False, loc_cols='location')
        dh.input_data = input_df
        dh.uncertainty_data = uncertainty_df
        dh.features = input_df.columns
        dh._load_data(existing_data=True)

        V, U = dh.get_data()

        assert V.shape == (2, 1)
        assert U.shape == (2, 1)
        assert dh.features == ['value']

    def test_get_data_drops_rows_with_nan_on_either_side(self):
        input_df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})
        uncertainty_df = pd.DataFrame({'value': [0.1, 0.2, np.nan]})
        dh = DataHandler.load_dataframe(input_df, uncertainty_df)

        V, U = dh.get_data()

        assert V.shape == (1, 1)
        assert U.shape == (1, 1)
        assert V[0, 0] == pytest.approx(1.0)
        assert U[0, 0] == pytest.approx(0.1)
