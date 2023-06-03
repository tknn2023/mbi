pub mod rpdiv;
pub mod knn_bruteforce;
pub mod distances;
pub mod nndescent;
pub mod heap;
pub mod trifree;
pub mod query;
pub mod knng;
pub mod vptree;
pub mod bf;
pub mod tknn_sf;
pub mod tknn_mbi;
pub mod tknn_mbi_lite;
pub mod tknn_sf_mbi;
pub mod stbf;

// use distances::*;
use rpdiv::*;
// use knn_bruteforce::*;

use pyo3::prelude::*;

// use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};



/// A Python module implemented in Rust.
#[pymodule]
fn knn_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tknn_mbi::TknnMBI>()?;
    m.add_class::<tknn_mbi_lite::TknnMBILite>()?;
    m.add_class::<knng::KNNGraph>()?;
    m.add_class::<bf::BF>()?;
    m.add_class::<tknn_sf::TknnSF>()?;
    m.add_class::<stbf::STBF>()?;

    Ok(())
}