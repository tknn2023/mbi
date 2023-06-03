use ndarray::prelude::*;
use super::rpdiv;
use fastrand;
use crate::heap::*;


pub fn nndescent(data: &[f32], dim: usize, k: usize, max_candidates: usize, max_iters: usize,
    rpdiv_iter: usize, rpdiv_size_limit: usize, dist: fn(&[f32], &[f32])->f32) -> (Array2<i32>, Array2<f32>) {

    let n_data = data.len();

    //create initial graph using rpdiv
    let (mut neighbors, mut distances) =
        rpdiv::rpdiv(&data, dim, k, rpdiv_iter, rpdiv_size_limit, dist);

    
    let neighbors_slice_mut = neighbors.as_slice_mut().unwrap();
    let distances_slice_mut = distances.as_slice_mut().unwrap();

    let mut new_flags = vec![vec![true;k]; n_data];

    let mut old_candidates = vec![vec![-1i32; max_candidates]; n_data];
    let mut new_candidates = vec![vec![-1i32; max_candidates]; n_data];
    let mut old_cand_size = vec![0usize; n_data];
    let mut new_cand_size = vec![0usize; n_data];

    for _iter in 0..max_iters {
        compute_candidates_arr(neighbors_slice_mut, k, &mut new_flags,
            &mut old_candidates, &mut new_candidates,
            &mut old_cand_size, &mut new_cand_size); // here

        let num_pushes = update_neighbors_arr(neighbors_slice_mut, distances_slice_mut, k, &mut new_flags,
            &old_candidates, &new_candidates,
            &old_cand_size, &new_cand_size, &data, dim, dist);

        // println!("{}: {}", iter, num_pushes); // disabled for tknn

        if num_pushes == 0 {
            break;
        }
    }

    return (neighbors, distances);

}

#[inline]
fn compute_candidates_arr(neighbors: &[i32], k:usize, new_flags: &mut [Vec<bool>],
                      old_candidates: &mut [Vec<i32>], new_candidates: &mut [Vec<i32>],
                      old_cand_size: &mut [usize], new_cand_size: &mut [usize]) {

    old_cand_size.fill(0);
    new_cand_size.fill(0);

    for (u, u_neighbors) in neighbors.chunks(k).enumerate(){
        for (v, is_new) in u_neighbors.iter().zip(new_flags[u].iter()) {
            if *v == -1i32 {continue;}
            if *is_new {
                reservoir_push(*v, &mut new_candidates[u], &mut new_cand_size[u]);
                reservoir_push(u as i32, &mut new_candidates[*v as usize], &mut new_cand_size[*v as usize]);
            }
            else{
                reservoir_push(*v, &mut old_candidates[u], &mut old_cand_size[u]);
                reservoir_push(u as i32, &mut old_candidates[*v as usize], &mut old_cand_size[*v as usize]);
            }
        }
    }
    
    // Should I use bitvec instead of the array of bools to increase the performance?
    let mut exist = vec![false; neighbors.len()/k];
    let max_cand_size = new_candidates[0].len();

    for u in 0..neighbors.len()/k {

        let u_new_candidates = &new_candidates[u];
        new_cand_size[u] = new_cand_size[u].min(max_cand_size);
        old_cand_size[u] = old_cand_size[u].min(max_cand_size);

        for i in 0..new_cand_size[u] {
            exist[u_new_candidates[i] as usize] = true;
        }

        for (v, f) in neighbors[u*k..u*k+k].iter().zip(new_flags[u].iter_mut()) { // here
            if *v == -1i32 {continue;}
            if exist[*v as usize] {
                *f = false;
            }
        }

        for i in 0..new_cand_size[u] {
            exist[u_new_candidates[i] as usize] = false;
        }
    }

}

#[inline]
fn update_neighbors_arr(neighbors: &mut [i32], distances: &mut [f32], k: usize, flags: &mut [Vec<bool>],
                    old_candidates: &[Vec<i32>], new_candidates: &[Vec<i32>],
                    old_cand_size: &[usize], new_cand_size: &[usize],
                    data: &[f32], dim: usize, dist: fn(&[f32], &[f32]) -> f32) -> usize{

    let mut num_pushes = 0usize;

    let n_data = data.len() / dim;

    for u in 0..n_data {

        let u_new_candidates = &new_candidates[u];
        let u_new_cand_size = new_cand_size[u];

        let u_old_candidates = &old_candidates[u];
        let u_old_cand_size = old_cand_size[u];
        
        for i in 0..u_new_cand_size {

            let v = u_new_candidates[i] as usize;
            
            for j in (i+1)..u_new_cand_size {

                let w = u_new_candidates[j] as usize;

                if v != w {
                    let d = dist(&data[v*dim..v*dim+dim], &data[w*dim..w*dim+dim]);
                    num_pushes += heap_push_no_dupl_flagged(w, d, &mut neighbors[v*k..v*k+k], &mut distances[v*k..v*k+k], &mut flags[v]);
                    num_pushes += heap_push_no_dupl_flagged(v, d, &mut neighbors[w*k..w*k+k], &mut distances[w*k..w*k+k], &mut flags[w]);
                }
            }

            for j in 0..u_old_cand_size {

                let w = u_old_candidates[j] as usize;

                if v != w {
                    let d = dist(&data[v*dim..v*dim+dim], &data[w*dim..w*dim+dim]);
                    num_pushes += heap_push_no_dupl_flagged(w, d, &mut neighbors[v*k..v*k+k], &mut distances[v*k..v*k+k], &mut flags[v]);
                    num_pushes += heap_push_no_dupl_flagged(v, d, &mut neighbors[w*k..w*k+k], &mut distances[w*k..w*k+k], &mut flags[w]);
                }
            }

        }
        
    }
    
    return num_pushes;
}

#[inline(always)]
fn reservoir_push(v: i32, candidates_u: &mut [i32], cand_size_u: &mut usize){
    if *cand_size_u < candidates_u.len() {
        candidates_u[*cand_size_u] = v;
    }
    else{
        let pos = fastrand::usize(..=*cand_size_u);
        if pos < candidates_u.len() {
            candidates_u[pos] = v;
        }
    }
    *cand_size_u += 1;
}


