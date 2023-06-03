use ndarray::prelude::*;
use crate::heap::heap_push;

pub fn knn_bruteforce(data_raw: &[f32], dim: usize, k: usize,
                dist: fn(&[f32], &[f32])->f32) -> Array2<i32> {

    let n_data = data_raw.len() / dim;

    let mut neighbors = Array::from_elem((n_data, k), -1i32);
    let mut distances = Array::from_elem((n_data, k), f32::INFINITY);
    
    let neigbhors_raw = neighbors.as_slice_mut().unwrap();
    let distances_raw = distances.as_slice_mut().unwrap();

    for (u, (neighbors_of_u, distances_to_u)) in neigbhors_raw.chunks_mut(k).zip(distances_raw.chunks_mut(k)).enumerate() {
        for v in 0..n_data {
            if u != v {
                let d = dist(&data_raw[u*dim..u*dim+dim], &data_raw[v*dim..v*dim+dim]);
                heap_push(v as i32, d, neighbors_of_u, distances_to_u);
            }
        }
    }

    return neighbors;

}

