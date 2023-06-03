use ndarray::{Array2, Array1};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;

use crate::heap::heap_push;
use crate::{distances::{*}};

#[inline(always)]
fn vec_of(v: usize, data: &[f32], dim: usize) -> &[f32] {
    return &data[v*dim..v*dim+dim];
}

#[pyclass]
pub struct STBF {
    data: Array2<f32>,
    timestamps: Array1<i64>,
    dist: String,
    k: usize,
    is_normed: bool
}

impl STBF {
    fn binary_search(&self, start_label: i64, end_label: i64) -> (usize, usize) {
        let mut lo = -1;
        let mut hi = self.data.shape()[0] as i32 - 1;
        while lo+1 < hi {
            let mid = ((lo + hi) / 2) as usize;
            if start_label <= self.timestamps[mid] {hi = mid as i32}
            else {lo = mid as i32}
        }
        let start_idx = hi as usize;

        let mut lo = 0;
        let mut hi = self.data.shape()[0] as i32;
        while lo+1 < hi {
            let mid = ((lo + hi) / 2) as usize;
            if end_label < self.timestamps[mid] {hi = mid as i32}
            else {lo = mid as i32}
        }
        let end_idx = hi as usize;
        return (start_idx, end_idx);
    }

    pub fn single_query(&mut self, q: &[f32], ts: i64, te: i64) -> (Vec<i32>, Vec<f32>){
        let data = self.data.as_slice().unwrap();
        let dim = self.data.shape()[1];

        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };

        let mut res_indices = vec![-1 ; self.k];
        let mut res_dists = vec![f32::INFINITY ; self.k];

        // let left = self.timestamps.partition_point(|&x| x < ts);
        // let right = self.timestamps.partition_point(|&x| x <= te);
        let (left, right) = self.binary_search(ts, te);

        for idx in left..right {
            heap_push(idx as i32, dist(q, vec_of(idx, data, dim)), &mut res_indices, &mut res_dists);
        }

        return (res_indices, res_dists);
    }
}

#[pymethods]
impl STBF {
    // Dump and load?

    // We assume that the data is ordered by timestamp.
    #[staticmethod]
    pub fn build<'py>(data: PyReadonlyArray2<f32>, timestamps: PyReadonlyArray1<i64>, dist: &str) -> Self {
        let is_normed = match dist {
            "normed_cosine" => true,
            "normed_squared_cosine" => true,
            _ => false
        };

        let tknn = STBF {
            data: data.as_array().to_owned(),
            // timestamps: timestamps.to_vec().unwrap(),
            timestamps: timestamps.as_array().to_owned(),
            dist: String::from(dist),
            k: 10,
            is_normed
        };
        
        return tknn;
    }

    #[pyo3(name="single_query")]
    fn py_single_query<'py>(&mut self, py: Python<'py>, q: PyReadonlyArray1<f32>, ts: i64, te: i64) -> (&'py PyArray1<i32>, &'py PyArray1<f32>) {
        let (res_indices, res_dists) =
            if self.is_normed {
                let q_slice = q.as_slice().unwrap();
                let q_denom = q_slice.iter().map(|x| *x * *x).sum::<f32>().sqrt();
                let q_vec: Vec<f32> = q_slice.iter().map(|x| *x/q_denom).collect::<Vec<f32>>();

                self.single_query(&q_vec, ts, te)
            } else {
                self.single_query(q.as_slice().unwrap(), ts, te)
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
