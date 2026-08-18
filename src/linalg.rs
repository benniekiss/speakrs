#[cfg(not(any(
    feature = "default-linalg",
    feature = "intel-mkl",
    feature = "openblas-static",
    feature = "openblas-system"
)))]
compile_error!(
    "speakrs requires a BLAS backend; enable default features or choose exactly one of `intel-mkl`, `openblas-static`, or `openblas-system`"
);

#[cfg(any(
    all(
        feature = "default-linalg",
        any(
            feature = "intel-mkl",
            feature = "openblas-static",
            feature = "openblas-system"
        )
    ),
    all(feature = "intel-mkl", feature = "openblas-static"),
    all(feature = "intel-mkl", feature = "openblas-system"),
    all(feature = "openblas-static", feature = "openblas-system")
))]
compile_error!(
    "speakrs supports only one BLAS backend; disable default features before enabling `intel-mkl`, `openblas-static`, or `openblas-system`"
);

#[cfg(all(feature = "intel-mkl", not(target_arch = "x86_64")))]
compile_error!("the `intel-mkl` feature is only supported on x86_64 targets");

#[cfg(feature = "intel-mkl")]
pub(crate) use ndarray_linalg_mkl::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(feature = "openblas-static")]
pub(crate) use ndarray_linalg_static::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(feature = "openblas-system")]
pub(crate) use ndarray_linalg_system::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(all(
    feature = "default-linalg",
    not(any(
        feature = "intel-mkl",
        feature = "openblas-static",
        feature = "openblas-system"
    ))
))]
pub(crate) use ndarray_linalg_default::{Eigh, Inverse, UPLO, error::LinalgError};

// `ndarray-linalg` intentionally leaves backend selection to its consumers.
// On macOS, keep the platform framework in the link graph for the default backend.
#[cfg(all(feature = "default-linalg", target_os = "macos"))]
use accelerate_src as _;

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Eigh, Inverse, UPLO};

    #[test]
    fn selected_backend_supports_required_operations() {
        let matrix = array![[2.0_f64, 1.0], [1.0, 2.0]];

        let (eigenvalues, _) = matrix.clone().eigh(UPLO::Lower).unwrap();
        let inverse = matrix.inv().unwrap();

        assert!((eigenvalues[0] - 1.0).abs() < 1e-12);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-12);
        assert!((inverse[[0, 0]] - 2.0 / 3.0).abs() < 1e-12);
        assert!((inverse[[0, 1]] + 1.0 / 3.0).abs() < 1e-12);
    }
}
