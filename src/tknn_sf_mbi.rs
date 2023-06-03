use ndarray::{Array2, Array1};
use pyo3::prelude::*;
use std::{time, vec};

use crate::distances::*;
use crate::nndescent::nndescent;
// use crate::trifree::{make_oriented_graph, remove_long_edges_eps, make_bidirectional_graph, make_trifree};
use crate::trifree::make_trifree;
use crate::heap::heap_push;
use crate::vptree::VPTree;
use crate::heap::{cand_pop, cand_push};
// use crate::rpdiv::rptree;

const VISITED: u8 = 1;
const UNVISITED: u8 = 0;

#[pyclass]
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone)]
pub struct TknnSF{
    start_idx: usize,
    end_idx: usize,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    search_tree: VPTree,
    visited: Vec<u8>,
    dist: String,
    is_normed: bool,
    dim: usize,
    k: usize,
    eps: f32,
    leaf_size: usize,
    cand_indices: Vec<i32>,
    cand_dists: Vec<f32>,

    // time
    time_tree_search: f64,
    time_query: f64,
}

#[inline(always)]
fn vec_of(v: usize, data: &[f32], dim: usize) -> &[f32] {
    return &data[v*dim..v*dim+dim];
}

impl TknnSF {

    fn tree_search(&mut self, data: &[f32], labels: &[i64], q: &[f32], ts: i64, te: i64, res_indices: &mut [i32], res_dists: &mut [f32]) -> usize {


        let dim = self.dim;
        let mut cnt = 1;

        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };

        let mut idx = 0usize;
        
        while self.search_tree.sizes[idx] > self.leaf_size {

            let v = self.search_tree.nodes[idx];
            let vt = labels[v as usize];
            let vd = dist(q, vec_of(v as usize, data, dim));
            
            cand_push(v, vd, &mut self.cand_indices, &mut self.cand_dists);
            
            if ts <= vt && vt <= te {
                heap_push(v, vd, res_indices, res_dists);
            }

            self.visited[v as usize] = VISITED;
            cnt += 1;

            if vd < self.search_tree.radii[idx] {
                idx = idx + 1;
            }
            else {
                idx += (self.search_tree.sizes[idx] + 1) / 2;
            }
        }

        for i in idx..idx+self.search_tree.sizes[idx] {
            let v = self.search_tree.nodes[i];
            let vt = labels[v as usize];
            let vd = dist(q, vec_of(v as usize, data, dim));

            cand_push(v, vd, &mut self.cand_indices, &mut self.cand_dists);

            if ts <= vt && vt <= te {
                heap_push(v, vd, res_indices, res_dists);
            }

            self.visited[v as usize] = VISITED;
            cnt += 1;
        }

        return cnt;
    }

    pub fn single_query(&mut self, data: &Array2<f32>, labels: &Array1<i64>, q: &[f32], ts: i64, te: i64) -> (Vec<i32>, Vec<f32>, usize) {

        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };
        
        let dim = self.dim;
        let data = &data.as_slice().unwrap()[self.start_idx*dim..self.end_idx*dim];
        let labels = &labels.as_slice().unwrap()[self.start_idx..self.end_idx];
        
        self.visited.fill(0);
        self.cand_indices.clear();
        self.cand_dists.clear();
        
        let mut res_indices = vec![-1 ; self.k];
        let mut res_dists = vec![f32::INFINITY ; self.k];
        
        // let t1 = time::Instant::now();
        let mut cnt = self.tree_search(data, labels, q, ts, te, &mut res_indices, &mut res_dists);
        // cnt = 0;
        // let t2 = time::Instant::now();
        
        while self.cand_indices.len() > 0 {

            let (u, d) = cand_pop(&mut self.cand_indices, &mut self.cand_dists);

            if d >= res_dists[0] * self.eps {
                break;
            }

            let s = self.indptr[u as usize] as usize;
            let e = self.indptr[u as usize + 1] as usize;
            for ind in s..e {
                let v = self.indices[ind] as usize;
                
                if self.visited[v] == UNVISITED {
                    let vt = labels[v];
                    let vd = dist(q, &data[v*dim..v*dim+dim]);
                    if vd < res_dists[0] * self.eps {
                        cand_push(v as i32, vd, &mut self.cand_indices, &mut self.cand_dists);
                        if ts <= vt && vt <= te {
                            heap_push(v as i32, vd, &mut res_indices, &mut res_dists);
                        }
                    }
                    self.visited[v] = VISITED;
                    cnt += 1;
                }
            }
        }
        // let t3 = time::Instant::now();

        // self.time_tree_search += (t2 - t1).as_secs_f64();
        // self.time_query += (t3 - t2).as_secs_f64();

        // println!("cnt: {}", cnt);
        return (res_indices, res_dists, cnt);
    }

    pub fn new(data: &Array2<f32>, start_idx: usize, end_idx: usize, n_neighbors: usize, max_candidates: usize,
        max_iters: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str) -> Self {
            
            let dim = data.shape()[1];
            let data_slice = &data.as_slice().unwrap()[start_idx*dim..end_idx*dim];
            let n_data = data.shape()[0];
            let is_normed = match dist {
                "normed_cosine" => true,
                "normed_squared_cosine" => true,
                _ => false
            };
            let distfn = match dist {
                "normed_cosine" => normed_cosine_distance,
                "normed_squared_cosine" => normed_squared_cosine_distance,
                _ => euclidean_distance
            };
    
            let s = time::SystemTime::now();
            let (neighbors, distances) = nndescent(data_slice, dim, n_neighbors, max_candidates, max_iters, rpdiv_iter, rpdiv_size_limit, distfn);
            let neighbors = neighbors.as_slice().unwrap();
            let distances = distances.as_slice().unwrap();
            println!("nndescent: {:?}", s.elapsed());
    
            let s = time::SystemTime::now();
            let csr = make_trifree(neighbors, distances, n_neighbors, tf_eps);
            println!("make_trifree: {:?}", s.elapsed());
            
            let s = time::SystemTime::now();
            let search_tree = VPTree::new(data_slice, dim, distfn);
            println!("build_tree: {:?}", s.elapsed());
            
            TknnSF{
                start_idx,
                end_idx,
                indptr: csr.indptr,
                indices: csr.indices,
                search_tree,
                visited: vec![0u8; n_data],
                dist: String::from(dist),
                is_normed,
                dim,
                k: 10,
                eps: 1.2,
                leaf_size: 10,
                cand_indices: vec![],
                cand_dists: vec![],
                time_tree_search: 0.0,
                time_query: 0.0,
            }
    }

    pub fn get_dummy() -> TknnSF {
        TknnSF{
            start_idx: 0,
            end_idx: 0,
            indptr: vec![],
            indices: vec![],
            search_tree: VPTree {
                nodes: vec![],
                radii: vec![],
                sizes: vec![],
            },
            visited: vec![0u8; 1],
            dist: "normed_cosine".to_string(),
            is_normed: true,
            dim: 1,
            k: 10,
            eps: 1.2,
            leaf_size: 10,
            cand_indices: vec![],
            cand_dists: vec![],
            time_tree_search: 0.0,
            time_query: 0.0,
        }
    }

    pub fn set_query_params(&mut self, k: usize, eps: f32, leaf_size: usize) {
        self.k = k;
        self.eps = eps;
        self.leaf_size = leaf_size;
    }
}

