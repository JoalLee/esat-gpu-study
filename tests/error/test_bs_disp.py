import esat.error.bs_disp as bs_disp_module
from esat.error.bs_disp import BSDISP


def test_parallel_disp_unpacks_args_and_passes_use_gpu(monkeypatch):
    captured = {}

    class FakeDisplacement:
        def __init__(self, sa, feature_labels, model_selected, threshold_dQ, max_search, features,
                     parallel, use_gpu):
            captured["sa"] = sa
            captured["feature_labels"] = feature_labels
            captured["model_selected"] = model_selected
            captured["threshold_dQ"] = threshold_dQ
            captured["max_search"] = max_search
            captured["features"] = features
            captured["parallel"] = parallel
            captured["use_gpu"] = use_gpu
            self.dQmax = None

        def run(self, batch):
            captured["batch"] = batch

    monkeypatch.setattr(bs_disp_module, "Displacement", FakeDisplacement)

    result = BSDISP._parallel_disp(("bs-1", object(), ["NO3"], 2, 0.1, 5, [0], [4, 2], True))

    assert result[0] == "bs-1"
    assert captured["batch"] == "bs-1"
    assert captured["parallel"] is False
    assert captured["use_gpu"] is True
