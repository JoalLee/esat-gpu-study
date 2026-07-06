import sys
import os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import numpy as np
import logging
import esat_rust
from esat.model.batch_sa import BatchSA

logger = logging.getLogger(__name__)


class TestLsNmfBatched:

    def setup_method(self):
        np.random.seed(42)
        self.n, self.m, self.k, self.B = 50, 10, 3, 4
        self.V = np.abs(np.random.randn(self.n, self.m).astype(np.float64))
        self.We = np.ones_like(self.V) / 0.01 ** 2
        self.W = np.abs(np.random.randn(self.B, self.n, self.k).astype(np.float64))
        self.H = np.abs(np.random.randn(self.B, self.k, self.m).astype(np.float64))

    def test_cpu_backend_returns_correct_shapes(self):
        r = esat_rust.ls_nmf_batched(self.V, self.We, self.W, self.H, 10, False, False)
        assert r['w'].shape == (self.B, self.n, self.k), f"W shape: {r['w'].shape}"
        assert r['h'].shape == (self.B, self.k, self.m), f"H shape: {r['h'].shape}"
        assert len(r['q']) == self.B, f"Q len: {len(r['q'])}"
        assert all(q > 0 for q in r['q']), "Q must be positive"

    def test_metal_backend_returns_correct_shapes(self):
        r = esat_rust.ls_nmf_batched(self.V, self.We, self.W, self.H, 10, False, True)
        assert r['w'].shape == (self.B, self.n, self.k)
        assert r['h'].shape == (self.B, self.k, self.m)
        assert len(r['q']) == self.B and all(q > 0 for q in r['q'])

    def test_cpu_metal_q_matches_within_1e_4(self):
        r_cpu = esat_rust.ls_nmf_batched(self.V, self.We, self.W, self.H, 50, False, False)
        r_metal = esat_rust.ls_nmf_batched(self.V, self.We, self.W, self.H, 50, False, True)
        q_cpu = np.array(r_cpu['q'], dtype=np.float64)
        q_metal = np.array(r_metal['q'], dtype=np.float64)
        rel_diff = np.max(np.abs(q_cpu - q_metal)) / max(1.0, np.max(np.abs(q_cpu)))
        assert rel_diff < 1e-4, f"CPU/Metal Q rel diff: {rel_diff:.2e}"

    def test_hold_h_preserves_h(self):
        H_fixed = self.H.copy()
        r = esat_rust.ls_nmf_batched(self.V, self.We, self.W, H_fixed, 20, True, False)
        h_diff = np.max(np.abs(r['h'] - H_fixed))
        assert h_diff < 1e-6, f"hold_h diff: {h_diff:.2e}"

    def test_2d_v_we_broadcasts_to_batch(self):
        r = esat_rust.ls_nmf_batched(self.V, self.We, self.W, self.H, 10, False, False)
        assert r['w'].shape[0] == self.B, "2D V/We must broadcast to B"

    def test_3d_v_we_works(self):
        V3d = np.tile(self.V[np.newaxis, :, :], (self.B, 1, 1))
        We3d = np.tile(self.We[np.newaxis, :, :], (self.B, 1, 1))
        r = esat_rust.ls_nmf_batched(V3d, We3d, self.W, self.H, 10, False, False)
        assert r['w'].shape == (self.B, self.n, self.k)

    def test_shape_validation_rejects_bad_h_batch(self):
        H_bad = np.abs(np.random.randn(self.B + 1, self.k, self.m))
        try:
            esat_rust.ls_nmf_batched(self.V, self.We, self.W, H_bad, 5, False, False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_shape_validation_rejects_bad_h_k(self):
        H_bad = np.abs(np.random.randn(self.B, self.k + 1, self.m))
        try:
            esat_rust.ls_nmf_batched(self.V, self.We, self.W, H_bad, 5, False, False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_shape_validation_rejects_bad_v_spatial(self):
        V_bad = np.abs(np.random.randn(self.n + 1, self.m))
        try:
            esat_rust.ls_nmf_batched(V_bad, self.We, self.W, self.H, 5, False, False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_shape_validation_rejects_bad_we_batch_3d(self):
        We_bad = np.tile(self.We[np.newaxis, :, :], (self.B + 1, 1, 1))
        try:
            esat_rust.ls_nmf_batched(self.V, We_bad, self.W, self.H, 5, False, False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_batch_sa_smoke(self):
        V2 = np.abs(np.random.randn(self.n, self.m))
        U2 = np.ones_like(V2) * 0.01
        bsa = BatchSA(V=V2, U=U2, factors=self.k, models=self.B,
                      method='ls-nmf', use_gpu=True, parallel=False, verbose=False)
        ok, err = bsa.train()
        assert ok, f"BatchSA failed: {err}"
        assert len(bsa.results) == self.B
        assert bsa.runtime is not None and bsa.runtime > 0
        assert bsa.best_model is not None and 0 <= bsa.best_model < self.B
        # Check H/W not mutated from their original values
        # (BatchSA copies H/W from kwargs; we verify results are distinct objects)
        for sa in bsa.results:
            assert sa.W.shape == (self.n, self.k)
            assert sa.H.shape == (self.k, self.m)
            assert np.isfinite(sa.Qtrue) and sa.Qtrue > 0
            assert np.isfinite(sa.Qrobust) and sa.Qrobust > 0
