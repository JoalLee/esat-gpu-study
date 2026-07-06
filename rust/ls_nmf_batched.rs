use crate::backend::select_device;
use candle_core::Tensor;
use numpy::{PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const EPS: f32 = 1e-9;

/// Batched LS-NMF on GPU.
/// V, We: 2D (n,m) or 3D (B,n,m). W: 3D (B,n,k), H: 3D (B,k,m).
/// If V/We are 2D they are broadcast across all B solves (avoids np.tile).
/// Returns {w: (B,n,k), h: (B,k,m), q: list[B]}.
#[pyfunction]
pub fn ls_nmf_batched<'py>(
    py: Python<'py>,
    v: PyReadonlyArrayDyn<f64>,
    we: PyReadonlyArrayDyn<f64>,
    w: PyReadonlyArrayDyn<f64>,
    h: PyReadonlyArrayDyn<f64>,
    max_iter: i32,
    hold_h: Option<bool>,
    prefer_gpu: Option<bool>,
) -> PyResult<PyObject> {
    let hold_h = hold_h.unwrap_or(false);
    let prefer_gpu = prefer_gpu.unwrap_or(true);
    let (device, device_name) = select_device(prefer_gpu);

    let v_ndim = v.as_array().ndim();
    let we_ndim = we.as_array().ndim();
    let w_arr = w.as_array().to_owned().into_dimensionality::<ndarray::Ix3>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("W must be 3D (B,n,k): {}", e))
    })?;
    let h_arr = h.as_array().to_owned().into_dimensionality::<ndarray::Ix3>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("H must be 3D (B,k,m): {}", e))
    })?;

    let b = w_arr.shape()[0];
    let n = w_arr.shape()[1];
    let m = h_arr.shape()[2];
    let k = w_arr.shape()[2];

    // Shape validation
    if we_ndim == 3 && we.as_array().shape()[0] != b {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "We batch dim {} != W batch dim {}", we.as_array().shape()[0], b
        )));
    }
    if h_arr.shape()[0] != b {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "H batch dim {} != W batch dim {}", h_arr.shape()[0], b
        )));
    }
    if h_arr.shape()[1] != k {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "H k-dim {} != W k-dim {}", h_arr.shape()[1], k
        )));
    }
    if v_ndim == 3 && v.as_array().shape()[0] != b {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("V batch dim {} != W batch dim {}", v.as_array().shape()[0], b)
        ));
    }
    if v_ndim >= 2 && (v.as_array().shape()[v_ndim - 2] != n || v.as_array().shape()[v_ndim - 1] != m) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("V spatial dims ({},{}) != expected ({},{})", v.as_array().shape()[v_ndim - 2], v.as_array().shape()[v_ndim - 1], n, m)
        ));
    }
    if we_ndim >= 2 && (we.as_array().shape()[we_ndim - 2] != n || we.as_array().shape()[we_ndim - 1] != m) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("We spatial dims ({},{}) != expected ({},{})", we.as_array().shape()[we_ndim - 2], we.as_array().shape()[we_ndim - 1], n, m)
        ));
    }

    // Helper: read f64 array into f32 tensor, handling 2D→3D conversion
    let make_t_2d_or_3d = |arr: ndarray::ArrayD<f64>, expected_ndim: usize, label: &str| -> PyResult<Tensor> {
        let actual_ndim = arr.ndim();
        let shape = if actual_ndim == 2 && expected_ndim == 3 {
            // Pad to (1, n, m) for broadcasting
            let (r, c) = (arr.shape()[0], arr.shape()[1]);
            vec![1usize, r, c]
        } else if actual_ndim == 3 {
            arr.shape().to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("{}: expected 2D or 3D, got {}D", label, actual_ndim)
            ));
        };
        let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
        Tensor::from_vec(data, shape.as_slice(), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", label, e)))
    };

    let v_t = make_t_2d_or_3d(v.as_array().to_owned(), 3, "V")?;
    let we_t = make_t_2d_or_3d(we.as_array().to_owned(), 3, "We")?;
    let mut w_t = make_t_2d_or_3d(w_arr.into_dyn(), 3, "W")?;
    let mut h_t = make_t_2d_or_3d(h_arr.into_dyn(), 3, "H")?;

    // Expand V, We to full batch size if they were 2D (repeat data B times)
    let v_t = if v_ndim == 2 {
        let v_data: Vec<f32> = v_t.flatten_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("V flat: {}", e)))?.to_vec1().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("V vec: {}", e)))?.into_iter().cycle().take(b * n * m).collect();
        Tensor::from_vec(v_data, &[b, n, m][..], &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("V expand: {}", e)))?
    } else {
        v_t
    };
    let we_t = if we_ndim == 2 {
        let we_data: Vec<f32> = we_t.flatten_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("We flat: {}", e)))?.to_vec1().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("We vec: {}", e)))?.into_iter().cycle().take(b * n * m).collect();
        Tensor::from_vec(we_data, &[b, n, m][..], &device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("We expand: {}", e)))?
    } else {
        we_t
    };

    let wev = we_t.mul(&v_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wev: {}", e)))?;

    // Batched multiplicative update
    for _ in 0..max_iter {
        // Update H (skip if hold_h)
        if !hold_h {
            let wh = w_t.matmul(&h_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wh: {}", e)))?;
            let wt = w_t.t().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wt: {}", e)))?;
            let h_num = wt.matmul(&wev).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h_num: {}", e)))?;
            let h_den = wt.matmul(&we_t.mul(&wh).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("we*wh: {}", e)))?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h_den: {}", e)))?
                .affine(1.0, EPS as f64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h_den eps: {}", e)))?;
            h_t = h_t.mul(&h_num.div(&h_den).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h_delta: {}", e)))?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h*mul: {}", e)))?;
        }

        // Update W
        let wh = w_t.matmul(&h_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wh2: {}", e)))?;
        let ht = h_t.t().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ht: {}", e)))?;
        let w_num = wev.matmul(&ht).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w_num: {}", e)))?;
        let w_den = we_t.mul(&wh).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("we*wh2: {}", e)))?
            .matmul(&ht).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w_den: {}", e)))?
            .affine(1.0, EPS as f64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w_den eps: {}", e)))?;
        w_t = w_t.mul(&w_num.div(&w_den).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w_delta: {}", e)))?)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w*mul: {}", e)))?;
    }

    // Compute per-slice Q = sum(We * (V - W@H)²)  (equivalent to sum(((V-WH)/U)²))
    let wh_final = w_t.matmul(&h_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wh_final: {}", e)))?;
    let residuals = v_t.sub(&wh_final).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("res: {}", e)))?;
    let weighted_res = we_t.mul(&residuals).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("wres: {}", e)))?;
    let q_per_slice = weighted_res.mul(&residuals)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("q mul: {}", e)))?
        .sum_keepdim(1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("q sum1: {}", e)))?
        .sum_keepdim(2).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("q sum2: {}", e)))?;
    let q_vec: Vec<f32> = q_per_slice.flatten_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("q flat: {}", e)))?.to_vec1()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("q vec: {}", e)))?;
    let q_py: Vec<f64> = q_vec.into_iter().map(|x| x as f64).collect();

    // Read back W, H (use flatten_all+to_vec1 instead of to_vec3 to avoid Metal panic)
    let w_out = {
        let w_flat_f32: Vec<f32> = w_t.flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w flatten: {}", e)))?
            .to_vec1()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w vec1: {}", e)))?;
        let w_f64: Vec<f64> = w_flat_f32.into_iter().map(|x: f32| x as f64).collect();
        let w_nd = ndarray::Array3::from_shape_vec((b, n, k), w_f64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("w shape: {}", e)))?;
        w_nd.to_pyarray(py).to_owned()
    };
    let h_out = {
        let h_flat_f32: Vec<f32> = h_t.flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h flatten: {}", e)))?
            .to_vec1()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h vec1: {}", e)))?;
        let h_f64: Vec<f64> = h_flat_f32.into_iter().map(|x: f32| x as f64).collect();
        let h_nd = ndarray::Array3::from_shape_vec((b, k, m), h_f64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("h shape: {}", e)))?;
        h_nd.to_pyarray(py).to_owned()
    };

    let result = PyDict::new(py);
    result.set_item("w", w_out).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("set w: {}", e)))?;
    result.set_item("h", h_out).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("set h: {}", e)))?;
    result.set_item("q", q_py).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("set q: {}", e)))?;
    result.set_item("backend", device_name).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("set backend: {}", e)))?;
    Ok(result.into())
}
