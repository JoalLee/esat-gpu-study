import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
import copy
import numpy as np
from esat.error.displacement import Displacement
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler
from esat.metrics import q_loss

logger = logging.getLogger(__name__)


class TestDisplacement:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    disp_name = "disp_test00"
    batch = None

    @classmethod
    def setup_class(self):
        logger.info("Running SA Test Setup")
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.input_file = os.path.join(self.data_path, "Dataset-BatonRouge-con.csv")
        self.uncertainty_file = os.path.join(self.data_path, "Dataset-BatonRouge-unc.csv")
        self.datahandler = DataHandler(
            input_path=self.input_file,
            uncertainty_path=self.uncertainty_file,
            index_col='Date'
        )
        self.V, self.U = self.datahandler.get_data()
        self.batch = BatchSA(V=self.V, U=self.U, models=2, factors=6, method="ls-nmf",
                             max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        self.batch.train()
        self.disp = None

    def test_run(self):
        selected_model = 1
        disp = Displacement(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                            features=[0])
        disp.run()
        assert disp.compiled_results is not None
        assert len(disp.increase_results) == 6

    def test_single_h_delta_q_matches_full_q_loss(self):
        selected_model = 1
        disp = Displacement(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                            features=[0])
        for factor_i in (0, 2, 5):
            for feature_j in (0, 3):
                for modifier in (0.25, 0.75, 1.5, 3.0):
                    new_H = copy.copy(disp.H)
                    new_value = disp.H[factor_i, feature_j] * modifier
                    new_H[factor_i, feature_j] = new_value
                    full_q = q_loss(V=disp.V, U=disp.U, W=disp.W, H=new_H)
                    delta_q = disp._q_for_h_value(factor_i, feature_j, new_value)
                    assert np.isclose(delta_q, full_q, rtol=1e-10, atol=1e-6)

    def test_save(self):
        selected_model = 1
        disp = Displacement(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                            features=[0])
        disp.run()
        save_path = os.path.join(self.data_path, "test_output")
        saved_file = disp.save(
            disp_name=self.disp_name,
            output_directory=save_path,
        )
        assert os.path.exists(str(os.path.join(save_path, f"{self.disp_name}.pkl")))

    def test_load(self):
        save_path = os.path.join(self.data_path, "test_output")
        save_file = os.path.join(save_path, f"{self.disp_name}.pkl")
        disp = Displacement.load(file_path=save_file)
        assert disp is not None
