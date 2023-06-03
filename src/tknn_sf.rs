use ndarray::{Array2, Array1};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;
use std::{time, vec};
use std::fs::File;
use std::io::prelude::*;

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
    data: Array2<f32>,
    timestamps: Vec<i64>,
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

    fn tree_search(&mut self, q: &[f32], ts: i64, te: i64, res_indices: &mut [i32], res_dists: &mut [f32]) -> usize {

        let data = self.data.as_slice().unwrap();
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
            let vt = self.timestamps[v as usize];
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
            let vt = self.timestamps[v as usize];
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

    pub fn single_query(&mut self, q: &[f32], ts: i64, te: i64) -> (Vec<i32>, Vec<f32>, usize) {

        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };
        
        let dim = self.dim;
        
        self.visited.fill(0);
        self.cand_indices.clear();
        self.cand_dists.clear();
        
        let mut res_indices = vec![-1 ; self.k];
        let mut res_dists = vec![f32::INFINITY ; self.k];
        
        // let t1 = time::Instant::now();
        let mut cnt = self.tree_search(q, ts, te, &mut res_indices, &mut res_dists);
        // cnt = 0;
        // let t2 = time::Instant::now();

        let data = self.data.as_slice().unwrap();
        
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
                    let vt = self.timestamps[v];
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

    pub fn new(data: Array2<f32>, timestamps: Array1<i64>, n_neighbors: usize, max_candidates: usize,
        max_iters: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str) -> Self {

            let data_slice = data.as_slice().unwrap();
            let (n_data, dim) = (data.shape()[0], data.shape()[1]);
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
                data: data.to_owned(),
                timestamps: timestamps.to_vec(),
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
        let data = Array2::zeros((0, 0));
        let timestamps = Vec::new();
        TknnSF{
            data,
            timestamps,
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
}


#[pymethods]
impl TknnSF {

    #[staticmethod]
    pub fn dump<'py>(knng: &TknnSF, path: &str){
        let mut file = File::create(path).unwrap();
        let encoded = bincode::serialize(knng).unwrap();
        file.write_all(&encoded).unwrap();
    }

    #[staticmethod]
    pub fn load<'py>(path: &str) -> Self{
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        let decoded: TknnSF = bincode::deserialize(&buffer[..]).unwrap();
        return decoded;
    }

    #[staticmethod]
    pub fn build_graph<'py>(data: PyReadonlyArray2<f32>, timestamps: PyReadonlyArray1<i64>, n_neighbors: usize, max_candidates: usize,
        max_iters: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str)
        -> Self{
        // let data_vec = data.to_vec().unwrap();
        let data_slice = data.as_slice().unwrap();
        let (n_data, dim) = (data.shape()[0], data.shape()[1]);
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

        // println!("ograph");
        // let ograph = make_oriented_graph(neighbors, distances, n_neighbors);
        // println!("tfgraph");
        // let tfgraph = remove_long_edges_eps(&ograph.0, &ograph.1, &ograph.2, tf_eps);
        // println!("bigraph");
        // let bigraph = make_bidirectional_graph(&tfgraph.0, &tfgraph.1, &tfgraph.2);


        let s = time::SystemTime::now();
        let csr = make_trifree(neighbors, distances, n_neighbors, tf_eps);
        println!("make_trifree: {:?}", s.elapsed());
        
        let s = time::SystemTime::now();
        let search_tree = VPTree::new(data_slice, dim, distfn);
        println!("build_tree: {:?}", s.elapsed());
        
        TknnSF{
            data: data.as_array().to_owned(),
            timestamps: timestamps.to_vec().unwrap(),
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

    pub fn set_query_params(&mut self, k: usize, eps: f32, leaf_size: usize) {
        self.k = k;
        self.eps = eps;
        self.leaf_size = leaf_size;
    }

    pub fn set_query_k(&mut self, k: usize) {
        self.k = k;
    }

    #[pyo3(name="single_query")]
    fn py_single_query<'py>(&mut self, py: Python<'py>, q: PyReadonlyArray1<f32>, ts: i64, te: i64) -> (&'py PyArray1<i32>, &'py PyArray1<f32>, usize){

        let (res_indices, res_dists, cnt_dist) =
            if self.is_normed {
                let q_slice = q.as_slice().unwrap();
                let q_denom = q_slice.iter().map(|x| *x * *x).sum::<f32>().sqrt();
                let q_vec: Vec<f32> = q_slice.iter().map(|x| *x/q_denom).collect::<Vec<f32>>();

                self.single_query(&q_vec, ts, te)
            }
            else{
                self.single_query(q.as_slice().unwrap(), ts, te)
            };

        return (
            PyArray1::from_vec(py, res_indices),
            PyArray1::from_vec(py, res_dists),
            cnt_dist
        );
    }

    pub fn get_time(&self) -> (f64, f64){
        (self.time_tree_search, self.time_query)
    }

    pub fn clear_time(&mut self) {
        self.time_tree_search = 0.0;
        self.time_query = 0.0;
    }
}

