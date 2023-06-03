use fastrand;
use ndarray::prelude::*;
use super::heap::{heap_push_no_dupl, heap_push_no_dupl_flagged};
use std::collections::HashSet;
use std::cmp::min;

pub fn rptree(data: &[f32], dim: usize, delta: f32, dist: fn(&[f32], &[f32]) -> f32) -> (Vec<i32>, Vec<i32>) {
    
    let n_data = data.len() / dim;

    let mut left = vec![-1; n_data];
    let mut right = vec![-1; n_data];
    
    let me = 0;
    let ids: Vec<usize> = (1..n_data).collect();

    rptree_rec(me, &ids, &mut left, &mut right, data, dim, delta, dist);
    
    return (left, right);

}

pub fn rptree_rec(me: usize, ids: &[usize], left: &mut [i32], right: &mut [i32], data: &[f32], dim: usize, delta: f32, dist: fn(&[f32], &[f32]) -> f32) {
    
    // println!("rpdiv! me:{}, size:{}", me, ids.len());

    let n_ids = ids.len();

    if n_ids == 0 {
        return;
    }
    if n_ids == 1 {
        left[me] = ids[0] as i32;
        return;
    }
    else if n_ids == 2 {
        left[me] = ids[0] as i32;
        right[me] = ids[1] as i32;
        return;
    }

    let mut cid: [usize; 2];
    let mut cvec: [&[f32]; 2];
    let mut csize: [usize; 2];
    let mut side = vec![false; n_ids];

    loop {
    
        cid = [ids[fastrand::usize(..n_ids)], ids[fastrand::usize(..n_ids)]];
        while cid[0] == cid[1] {
            cid[1] = ids[fastrand::usize(..n_ids)];
        }

        cvec = [&data[cid[0]*dim..cid[0]*dim+dim], &data[cid[1]*dim..cid[1]*dim+dim]];
        csize = [0usize,0];

        for (i, id) in ids.iter().enumerate() {

            let thisvec = &data[*id*dim..*id*dim+dim];

            let d = [dist(cvec[0], thisvec), dist(cvec[1], thisvec)];

            let w = if (d[0] - d[1]).abs() <= f32::EPSILON {
                fastrand::usize(..2)
            } else if d[0] < d[1] {
                0
            } else {
                1
            };

            csize[w] += 1;
            side[i] = w != 0;
        }

        csize[0] -= 1;
        csize[1] -= 1;

        let large = csize[0].max(csize[1]) as f32;
        let small = csize[0].min(csize[1]) as f32;

        if n_ids < 20 {
            if large < small + 10.0 {
                break;
            } else {
                // println!("Unbalanced, retry. left:{}, right:{}", csize[0], csize[1]);
            }
        } else if large < delta * small {
            break;
        } else {
            // println!("Unbalanced, retry. left:{}, right:{} (delta: {}/{})",
            //         csize[0], csize[1], large/small, delta);
        }

    }

    let mut parts: [Vec<usize>; 2] = [Vec::with_capacity(csize[0]), Vec::with_capacity(csize[1])];

    for (is_right, u) in side.iter().zip(ids){
        let w = *is_right as usize;
        if *u != cid[w] {
            parts[w].push(*u);
        }
    }

    left[me] = cid[0] as i32;
    right[me] = cid[1] as i32;


    rptree_rec(cid[0], &parts[0], left, right, data, dim, delta, dist);
    rptree_rec(cid[1], &parts[1], left, right, data, dim, delta, dist);

}


pub fn rpdiv(data: &[f32], dim: usize, k: usize, n_iter: usize, size_limit: usize, dist: fn(&[f32], &[f32]) -> f32) -> (Array2<i32>, Array2<f32>){
    
    let n_data = data.len() / dim;

    let mut neighbors = Array::from_elem((n_data, k), -1i32);
    let mut distances = Array::from_elem((n_data, k), f32::INFINITY);

    let neighbors_slice_mut = neighbors.as_slice_mut().unwrap();
    let distances_slice_mut = distances.as_slice_mut().unwrap();
    
    for _iter in 0..n_iter {
        let _n_push_hits = partition_and_update_graph(data, dim, k, neighbors_slice_mut, distances_slice_mut, size_limit, dist);
        // println!("{} {}", iter, n_push_hits); // disabled for tknn
    }

    return (neighbors, distances);
}

pub fn rpdiv_nnd(data: &[f32], dim: usize, k: usize, n_iter: usize, size_limit: usize, dist: fn(&[f32], &[f32]) -> f32, nnd_iter: usize, max_candidates: usize) -> (Array2<i32>, Array2<f32>){
    
    let n_data = data.len() / dim;

    let mut neighbors = Array::from_elem((n_data, k), -1i32);
    let mut distances = Array::from_elem((n_data, k), f32::INFINITY);

    let neighbors_slice_mut = neighbors.as_slice_mut().unwrap();
    let distances_slice_mut = distances.as_slice_mut().unwrap();
    
    for _iter in 0..n_iter {
        let _n_push_hits = partition_and_update_graph_nnd(data, dim, k, neighbors_slice_mut, distances_slice_mut, size_limit, dist, nnd_iter, max_candidates);
        println!("{} {} !", _iter, _n_push_hits); // disabled for tknn
    }
    println!("ho");

    return (neighbors, distances);
}

pub fn partition_and_update_graph(data: &[f32], dim: usize, k: usize, neighbors: &mut [i32],
    distances: &mut [f32], size_limit: usize, dist: fn(&[f32], &[f32]) -> f32) -> usize{

    let mut blocks = Vec::<Vec<usize>>::new();
    recursive_partition(data, dim, (0..(data.len()/dim)).collect(), &mut blocks, size_limit, dist);

    let mut n_push_hits = 0usize;

    for block in blocks {
        for u in &block {
            let neighbors_of_u = &mut neighbors[(*u)*k..(*u)*k+k];
            let distances_to_u = &mut distances[(*u)*k..(*u)*k+k];
            
            for v in &block {
                if u != v {
                    let d = dist(&data[(*u)*dim..(*u)*dim+dim], &data[(*v)*dim..(*v)*dim+dim]);
                    n_push_hits += heap_push_no_dupl(*v, d, neighbors_of_u, distances_to_u);
                }
            }
        }
    }

    return n_push_hits;
}

fn random_neighbors(data: &[f32], dim: usize, k: usize, neighbors: &mut [i32], distances: &mut [f32], flags: &mut [Vec<bool>], dist: fn(&[f32], &[f32]) -> f32) {
    let n_data = data.len()/dim;
    let rng = fastrand::Rng::new();
    let n_neighbors = min(n_data-1, k);

    for u in 0..n_data {
        let mut pushed_neighbors = HashSet::<usize>::new();
        while pushed_neighbors.len() < n_neighbors {
            let v = rng.usize(..n_data);
            if v == u { continue; }
            pushed_neighbors.insert(v);
            let d = dist(&data[u*dim..u*dim+dim], &data[v*dim..v*dim+dim]);
            heap_push_no_dupl_flagged(v, d, &mut neighbors[u*k..u*k+k], &mut distances[u*k..u*k+k], flags[u].as_mut_slice());
            // n_push_hit += heap_push_no_dupl_flagged(u, d, &mut neighbors[v*k..v*k+k], &mut distances[v*k..v*k+k], flags[v].as_mut_slice());
        }
    }
}

pub fn partition_and_update_graph_nnd(data: &[f32], dim: usize, k: usize, neighbors: &mut [i32],
    distances: &mut [f32], size_limit: usize, dist: fn(&[f32], &[f32]) -> f32, nnd_iter: usize, max_candidates: usize) -> usize{
    // println!("!");
    let mut blocks = Vec::<Vec<usize>>::new();
    recursive_partition(data, dim, (0..(data.len()/dim)).collect(), &mut blocks, size_limit, dist);
    // println!("!!");
    let n_data = data.len() / dim;

    let mut n_push_hits = 0usize;

    // println!("!!!!");
    let mut old_candidates = vec![vec![-1i32; max_candidates]; n_data];
    let mut new_candidates = vec![vec![-1i32; max_candidates]; n_data];
    let mut old_cand_size = vec![0usize; n_data];
    let mut new_cand_size = vec![0usize; n_data];
    let mut new_flag = vec![vec![true;k]; n_data];

    for block in blocks {
        let n_block = block.len();
        // Convert n_data to n_block
        let mut sub_data_vec = vec![0f32; n_block*dim];
        let sub_data = sub_data_vec.as_mut_slice();
        let mut sub_neighbors_vec = vec![-1i32; n_block*k];
        let sub_neighbors = sub_neighbors_vec.as_mut_slice();
        let mut sub_distances_vec = vec![f32::INFINITY; n_block*k];
        let sub_distances = &mut sub_distances_vec.as_mut_slice();
        let mut sub_flag = &mut vec![vec![true;k]; n_data];

        let mut sub_old_candidates = vec![vec![-1i32; max_candidates]; n_block];
        let mut sub_new_candidates = vec![vec![-1i32; max_candidates]; n_block];
        let mut sub_old_cand_size = vec![0usize; n_block];
        let mut sub_new_cand_size = vec![0usize; n_block];
        let mut origin_idx = vec![0usize; n_block];
        let mut dense_idx = vec![-1i32; n_data];
        for idx in 0..n_block {
            let u = block[idx];
            origin_idx[idx] = u;
            dense_idx[u] = idx as i32;
        }
        for idx in 0..n_block {
            let u = block[idx];
            for d in 0..dim {
                sub_data[idx*dim+d] = data[u*dim+d];
            }
            // candidates
            for i in 0..old_cand_size[u] {
                let v = old_candidates[u][i];
                if v == -1 {
                    sub_old_candidates[idx][i] = -1;
                } else {
                    sub_old_candidates[idx][i] = dense_idx[v as usize];
                }
            }
            sub_old_cand_size[idx] = old_cand_size[u];
            for i in 0..new_cand_size[u] {
                let v = new_candidates[u][i];
                if v == -1 {
                    sub_new_candidates[idx][i] = -1;
                } else {
                    sub_new_candidates[idx][i] = dense_idx[v as usize];
                }
            }
            sub_new_cand_size[idx] = new_cand_size[u];
            for i in 0..k {
                if neighbors[u*k+i] == -1 {
                    sub_neighbors[idx*k+i] = -1;
                    sub_distances[idx*k+i] = f32::INFINITY;
                    continue;
                }
                sub_neighbors[idx*k+i] = dense_idx[neighbors[(u*k+i) as usize] as usize] as i32;
                sub_distances[idx*k+i] = distances[u*k+i];
            }
            // flag
            for i in 0..k {
                sub_flag[idx][i] = new_flag[u][i];
            }
        }

        // println!("!!!!!");
        // 이 부분에서, 원래 블럭 내에서 bf로 그래프 만들기였는데 이제는 랜덤 후 nnd로 만들기
        random_neighbors(sub_data, dim, k, sub_neighbors, sub_distances, sub_flag, dist);
        // println!("!!!!!!");
        for _iter in 0..nnd_iter {
            compute_candidates_arr(sub_neighbors, k, &mut sub_flag,
                &mut sub_old_candidates, &mut sub_new_candidates,
                &mut sub_old_cand_size, &mut sub_new_cand_size); // here
    
            let num_pushes = update_neighbors_arr(sub_neighbors, sub_distances, k, &mut sub_flag,
                &sub_old_candidates, &sub_new_candidates,
                &sub_old_cand_size, &sub_new_cand_size, &sub_data, dim, dist);
    
            println!("{} {} !!", _iter, num_pushes); // disabled for tknn
    
            if num_pushes == 0 {
                break;
            }
            n_push_hits += num_pushes;
        }
        // println!("!!!!!!!");
        for idx in 0..n_block {
            let u = origin_idx[idx];
            for i in 0..k {
                if sub_neighbors[idx*k+i] == -1 {
                    neighbors[u*k+i] = -1;
                    distances[u*k+i] = f32::INFINITY;
                    continue;
                }
                neighbors[u*k+i] = origin_idx[sub_neighbors[(idx*k+i) as usize] as usize] as i32;
                distances[u*k+i] = sub_distances[idx*k+i];
            }
            for i in 0..sub_old_cand_size[idx] {
                let v = sub_old_candidates[idx][i];
                if v == -1 {
                    old_candidates[u][i] = -1;
                } else {
                    old_candidates[u][i] = origin_idx[v as usize] as i32;
                }
            }
            old_cand_size[u] = sub_old_cand_size[idx];
            for i in 0..sub_new_cand_size[idx] {
                let v = sub_new_candidates[idx][i];
                if v == -1 {
                    new_candidates[u][i] = -1;
                } else {
                    new_candidates[u][i] = origin_idx[v as usize] as i32;
                }
            }
            new_cand_size[u] = sub_new_cand_size[idx];
            // flag
            for i in 0..k {
                new_flag[u][i] = sub_flag[idx][i];
            }
        }
    }

    return n_push_hits;
}



pub fn partition(data: &[f32], dim: usize, ids: &[usize], dist: fn(&[f32], &[f32]) -> f32) -> (Vec<usize>, Vec<usize>) {
    
    let mut x = [fastrand::usize(..ids.len()), fastrand::usize(..ids.len())];
    while x[0] == x[1] {
        x[1] = fastrand::usize(..ids.len());
    }
    
    let c = [&data[ids[x[0]]*dim..ids[x[0]]*dim + dim], &data[ids[x[1]]*dim..ids[x[1]]*dim+dim]];


    let mut c_size = [0usize, 0usize];
    let mut side = vec![false; ids.len()];

    for (i, u) in ids.iter().enumerate() {
        let uvec = &data[(*u)*dim..(*u)*dim+dim];
        let d = [dist(uvec, c[0]), dist(uvec, c[1])];

        let w: usize =  if (d[0] - d[1]).abs() <= f32::EPSILON { fastrand::usize(..2) }
                        else{ (d[0] > d[1]) as usize };
        c_size[w] += 1;
        side[i] = w == 1;
    }

    let mut block = (Vec::<usize>::with_capacity(c_size[0]), Vec::<usize>::with_capacity(c_size[1]));

    for (is_right, u) in side.iter().zip(ids){
        if *is_right {
            block.1.push(*u);
        }
        else{
            block.0.push(*u);
        }
    }
    
    return block;
    
}


pub fn recursive_partition(data: &[f32], dim: usize, ids: Vec<usize>,
    blocks: &mut Vec<Vec<usize>>, size_limit: usize, dist: fn(&[f32], &[f32]) -> f32){
    
    if ids.len() < size_limit {
        blocks.push(ids);
    }
    else{
        let (left_ids, right_ids) = partition(data, dim, &ids, dist);
        recursive_partition(data, dim, left_ids, blocks, size_limit, dist);
        recursive_partition(data, dim, right_ids, blocks, size_limit, dist);
    }
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