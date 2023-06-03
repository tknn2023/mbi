use ndarray::{Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;

use crate::heap::heap_push;
use crate::{distances::{*}};

#[inline(always)]
fn vec_of(v: usize, data: &[f32], dim: usize) -> &[f32] {
    return &data[v*dim..v*dim+dim];
}

#[pyclass]
pub struct BF {
    data: Array2<f32>,

    // parameters for knng
    dist: String,
    k: usize,
    is_normed: bool
}

impl BF {
    pub fn new(data: Array2<f32>, dist: &str) -> BF {
        let is_normed = match dist {
            "normed_cosine" => true,
            "normed_squared_cosine" => true,
            _ => false
        };
        BF {
            data: data,
            dist: String::from(dist),
            k: 10,
            is_normed: is_normed,
        }
    }

    pub fn single_query(&mut self, q: &[f32]) -> (Vec<i32>, Vec<f32>){
        let data = self.data.as_slice().unwrap();
        let dim = self.data.shape()[1];

        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };

        let mut res_indices = vec![-1 ; self.k];
        let mut res_dists = vec![f32::INFINITY ; self.k];

        for idx in 0..self.data.shape()[0] {
            heap_push(idx as i32, dist(q, vec_of(idx, data, dim)), &mut res_indices, &mut res_dists);
        }

        return (res_indices, res_dists);
    }
}

#[pymethods]
impl BF {
    // Dump and load?

    #[staticmethod]
    pub fn build<'py>(data: PyReadonlyArray2<f32>, dist: &str) -> Self {
        let is_normed = match dist {
            "normed_cosine" => true,
            "normed_squared_cosine" => true,
            _ => false
        };

        let data_array = data.as_array().to_owned();

        let tknn = BF {
            data: data_array,
            dist: String::from(dist),
            k: 10,
            is_normed
        };
        
        return tknn;
    }

    #[pyo3(name="single_query")]
    fn py_single_query<'py>(&mut self, py: Python<'py>, q: PyReadonlyArray1<f32>) -> (&'py PyArray1<i32>, &'py PyArray1<f32>) {
        let (res_indices, res_dists) =
            if self.is_normed {
                let q_slice = q.as_slice().unwrap();
                let q_denom = q_slice.iter().map(|x| *x * *x).sum::<f32>().sqrt();
                let q_vec: Vec<f32> = q_slice.iter().map(|x| *x/q_denom).collect::<Vec<f32>>();

                self.single_query(&q_vec)
            } else {
                self.single_query(q.as_slice().unwrap())
            };
        
        return (
            PyArray1::from_vec(py, res_indices),
            PyArray1::from_vec(py, res_dists)
        );
    }

    pub fn set_query_params(&mut self, k: usize) {
        self.k = k;
    }
}
