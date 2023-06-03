use ndarray::{Array2, Array1, concatenate, Axis, s};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1};
use std::cmp::{min, max};
use std::time;
use std::fs::File;
use pyo3::prelude::*;
use std::io::prelude::*;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::{tknn_sf::TknnSF, distances::{*}, heap::heap_push};

const KNN: u8 = 0;
const BF: u8 = 1;

const INF: usize = 1 << 24;

#[inline(always)]
fn vec_of(v: usize, data: &[f32], dim: usize) -> &[f32] {
    return &data[v*dim..v*dim+dim];
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct KNNode {
    idx: usize,
    start_idx: usize,
    end_idx: usize,
    level: usize,
    complete: bool,
    dummy: bool,
    knng: TknnSF,
}

impl KNNode {
    fn new(idx:usize, start_idx: usize, end_idx: usize, level: usize, complete: bool, dummy: bool, knng: TknnSF) -> KNNode {
        KNNode {
            idx,
            start_idx,
            end_idx,
            level,
            complete,
            dummy,
            knng
        }
    }

    pub fn print(&self) {
        print!("[{:?}, {:?}]", self.start_idx, self.end_idx);
    }
}

#[inline(always)]
fn has_left_sibling(idx: usize, level: usize) -> bool {
    return idx >= ((1 << (level+1)) - 1);
}

#[inline(always)]
fn get_left_sibling(idx: usize, level: usize) -> usize {
    return idx - ((1 << (level+1)) - 1);
}

#[inline(always)]
fn get_right_sibling(idx: usize, level: usize) -> usize {
    return idx + ((1 << (level+1)) - 1);
}

#[pyclass]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TknnMBI {
    blocks: Vec<KNNode>,
    root_idx: usize,
    growing_idx: usize,
    data: Array2<f32>,
    label: Array1<i64>,
    tknn_leaf_size: usize,
    threshold: f32, // 블럭의 전체 기간(인덱스)에 대해 탐색 기간(인덱스)의 비율이 threshold 이상이면 탐색, 1이면 완전히 포함될 때만 탐색

    // parameters for knng
    n_neighbors: usize,
    max_candidates: usize,
    max_iter: usize,
    rpdiv_iter: usize,
    rpdiv_size_limit: usize,
    tf_eps: f32,
    dist: String,
    k: usize,
    is_normed: bool,

    bf_limit: usize,

    // time
    time_get_query: f64,
    time_query: f64,
    time_refine_result: f64,
}

fn sorted_merge(a: &[(i32, f32)], b: &[(i32, f32)], k: usize) -> Vec<(i32, f32)> {
    let mut result = Vec::new();
    result.reserve(k);
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() && result.len() < k {
        if a[i].1 < b[j].1 {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }
    while i < a.len() && result.len() < k {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() && result.len() < k {
        result.push(b[j]);
        j += 1;
    }
    return result;
}

impl TknnMBI {
    // fn get_knng(&self, data: Array2<f32>) -> TknnSF {

    fn reserve(&mut self, n: usize) {
        let n = (n+self.tknn_leaf_size-1) / self.tknn_leaf_size;
        let mut target_len = 1;
        while target_len < n {
            target_len *= 2;
        }
        let mut level = 0;
        for block_idx in self.blocks.len()..(2*target_len-1) {
            let block = KNNode::new(block_idx, INF, INF, level, false, true, TknnSF::get_dummy());
            self.blocks.push(block);
            let sibling_idx = get_left_sibling(block_idx, level);
            level = if has_left_sibling(block_idx, level) && self.blocks[sibling_idx].level == level {level + 1} else {0}
        }
        self.root_idx = self.blocks.len()-1;
    }

    fn add_node(&mut self, start_idx: usize, end_idx: usize, complete: bool, level: usize) {
        let block_idx = self.growing_idx;
        self.blocks[block_idx].idx = block_idx;
        self.blocks[block_idx].start_idx = start_idx;
        self.blocks[block_idx].end_idx = end_idx;
        self.blocks[block_idx].level = level;
        self.blocks[block_idx].complete = complete;
        self.blocks[block_idx].dummy = false;

        if level == 0 {
            let mut now_idx = block_idx;
            while now_idx < self.root_idx && self.blocks[now_idx+1].level == self.blocks[now_idx].level + 1 {
                now_idx += 1;
                if self.blocks[now_idx].dummy { self.blocks[now_idx].end_idx = end_idx; }
            }
            now_idx = block_idx;
            let mut sibling_idx = get_right_sibling(now_idx, self.blocks[now_idx].level);
            while sibling_idx < self.root_idx && self.blocks[sibling_idx].level == self.blocks[now_idx].level && self.blocks[sibling_idx+1].level == self.blocks[now_idx].level+1 {
                self.blocks[sibling_idx+1].start_idx = self.blocks[now_idx].start_idx;
                now_idx = sibling_idx + 1;
                sibling_idx = get_right_sibling(now_idx, self.blocks[now_idx].level);
            }
        }

        if complete {
            let knng_data = self.data.slice(s![self.blocks[block_idx].start_idx as usize..self.blocks[block_idx].end_idx as usize+1, ..]).to_owned();
            let knng_label = self.label.slice(s![self.blocks[block_idx].start_idx as usize..self.blocks[block_idx].end_idx as usize+1]).to_owned();
            let mut knng = TknnSF::new(knng_data, knng_label, self.n_neighbors, self.max_candidates, self.max_iter, self.rpdiv_iter, self.rpdiv_size_limit, self.tf_eps, self.dist.as_str());
            knng.set_query_params(self.k, 1.4, 10);
            self.blocks[block_idx].knng = knng;

            self.growing_idx += 1;
            if has_left_sibling(block_idx, level) && self.blocks[get_left_sibling(block_idx, level)].level == level {
                self.add_node(self.blocks[get_left_sibling(block_idx, level)].start_idx, end_idx, complete, level+1);
            }
        }
    }

    fn add_data(&mut self, data: Array2<f32>, label: Array1<i64>) {
        let old_len = self.data.shape()[0];
        
        if self.data.shape()[0] == 0 {
            self.data = data;
            self.label = label;
        } else {
            self.data = concatenate(Axis(0), &[self.data.view(), data.view()]).unwrap();
            self.label = concatenate(Axis(0), &[self.label.view(), label.view()]).unwrap();
        }

        let new_len = self.data.shape()[0];
        self.reserve(new_len);

        for i in (old_len/self.tknn_leaf_size)..(new_len/self.tknn_leaf_size) {
            println!("add node: {} / {}", i, new_len/self.tknn_leaf_size);
            self.add_node(i * self.tknn_leaf_size, (i+1) * self.tknn_leaf_size - 1, true, 0);
        }

        if new_len % self.tknn_leaf_size != 0 {
            println!("add node: growing node");
            self.add_node((new_len/self.tknn_leaf_size) * self.tknn_leaf_size, new_len-1, false, 0);
        }
    }

    fn add_node_parallel(&mut self, start_idx: usize, end_idx: usize, complete: bool, level: usize, forbuild: &mut Vec<(usize, usize, usize)>) {
        let block_idx = self.growing_idx;
        self.blocks[block_idx].idx = block_idx;
        self.blocks[block_idx].start_idx = start_idx;
        self.blocks[block_idx].end_idx = end_idx;
        self.blocks[block_idx].level = level;
        self.blocks[block_idx].complete = complete;
        self.blocks[block_idx].dummy = false;

        if level == 0 {
            let mut now_idx = block_idx;
            while now_idx < self.root_idx && self.blocks[now_idx+1].level == self.blocks[now_idx].level + 1 {
                now_idx += 1;
                if self.blocks[now_idx].dummy { self.blocks[now_idx].end_idx = end_idx; }
            }
            now_idx = block_idx;
            let mut sibling_idx = get_right_sibling(now_idx, self.blocks[now_idx].level);
            while sibling_idx < self.root_idx && self.blocks[sibling_idx].level == self.blocks[now_idx].level && self.blocks[sibling_idx+1].level == self.blocks[now_idx].level+1 {
                self.blocks[sibling_idx+1].start_idx = self.blocks[now_idx].start_idx;
                now_idx = sibling_idx + 1;
                sibling_idx = get_right_sibling(now_idx, self.blocks[now_idx].level);
            }
        }

        if complete {
            // // 이 부분을, 여기서 할 게 아니고 넘겨줘야 함
            // let knng_data = self.data.slice(s![self.blocks[block_idx].start_idx as usize..self.blocks[block_idx].end_idx as usize+1, ..]).to_owned();
            // let knng_label = self.label.slice(s![self.blocks[block_idx].start_idx as usize..self.blocks[block_idx].end_idx as usize+1]).to_owned();
            // let mut knng = TknnSF::new(knng_data, knng_label, self.n_neighbors, self.max_candidates, self.max_iter, self.rpdiv_iter, self.rpdiv_size_limit, self.tf_eps, self.dist.as_str());
            // knng.set_query_params(self.k, 1.4, 10);
            // self.blocks[block_idx].knng = knng;
            // //
            self.blocks[block_idx].knng = TknnSF::get_dummy();
            forbuild.push((block_idx, self.blocks[block_idx].start_idx, self.blocks[block_idx].end_idx));

            self.growing_idx += 1;
            if has_left_sibling(block_idx, level) && self.blocks[get_left_sibling(block_idx, level)].level == level {
                self.add_node_parallel(self.blocks[get_left_sibling(block_idx, level)].start_idx, end_idx, complete, level+1, forbuild);
            }
        }
    }

    fn add_data_parallel(&mut self, data: Array2<f32>, label: Array1<i64>) {
        let old_len = self.data.shape()[0];
        
        if self.data.shape()[0] == 0 {
            self.data = data;
            self.label = label;
        } else {
            self.data = concatenate(Axis(0), &[self.data.view(), data.view()]).unwrap();
            self.label = concatenate(Axis(0), &[self.label.view(), label.view()]).unwrap();
        }

        let new_len = self.data.shape()[0];
        self.reserve(new_len);
        
        let mut forbuild = Vec::new();
        
        for i in (old_len/self.tknn_leaf_size)..(new_len/self.tknn_leaf_size) {
            println!("add node: {} / {}", i, new_len/self.tknn_leaf_size);
            self.add_node_parallel(i * self.tknn_leaf_size, (i+1) * self.tknn_leaf_size - 1, true, 0, &mut forbuild);
        }

        if new_len % self.tknn_leaf_size != 0 {
            println!("add node: growing node");
            self.add_node_parallel((new_len/self.tknn_leaf_size) * self.tknn_leaf_size, new_len-1, false, 0, &mut forbuild);
        }

        let data_for_share = self.data.clone();
        let label_for_share = self.label.clone();
        let shared_data: Arc<Array2<f32>> = Arc::new(data_for_share);
        let shared_label: Arc<Array1<i64>> = Arc::new(label_for_share);
        
        let shared_result = Arc::new(Mutex::new(Vec::new()));

        if forbuild.len() >= 2 { // 조절 가능

            let threads: Vec<_> = (0..forbuild.len()).map(|i| {
                let data = shared_data.clone().to_owned();
                let label = shared_label.clone().to_owned();
                let (block_idx, start_idx, end_idx) = forbuild[i];
                let n_neighbors = self.n_neighbors;
                let max_candidates = self.max_candidates;
                let max_iter = self.max_iter;
                let rpdiv_iter = self.rpdiv_iter;
                let rpdiv_size_limit = self.rpdiv_size_limit;
                let tf_eps = self.tf_eps;
                let dist = self.dist.clone();
                let k = self.k;
                let result = shared_result.clone();
                thread::spawn(move || {
                    let knng_data = data.slice(s![start_idx as usize..end_idx as usize+1, ..]).to_owned();
                    let knng_label = label.slice(s![start_idx as usize..end_idx as usize+1]).to_owned();
                    let mut knng = TknnSF::new(knng_data, knng_label, n_neighbors, max_candidates, max_iter, rpdiv_iter, rpdiv_size_limit, tf_eps, dist.as_str());
                    knng.set_query_params(k, 1.4, 10);

                    let mut res = result.lock().unwrap();
                    res.push((block_idx, knng));
                })
            }).collect();

            for thread in threads {
                thread.join().unwrap();
            }
            
            for (block_idx, knng) in shared_result.lock().unwrap().iter() {
                self.blocks[*block_idx].knng = knng.clone();
            }
        } else if forbuild.len() == 1 {
            let (block_idx, start_idx, end_idx) = forbuild[0];
            let knng_data = self.data.slice(s![start_idx as usize..end_idx as usize+1, ..]).to_owned();
            let knng_label = self.label.slice(s![start_idx as usize..end_idx as usize+1]).to_owned();
            let mut knng = TknnSF::new(knng_data, knng_label, self.n_neighbors, self.max_candidates, self.max_iter, self.rpdiv_iter, self.rpdiv_size_limit, self.tf_eps, self.dist.as_str());
            knng.set_query_params(self.k, 1.4, 10);
            self.blocks[block_idx].knng = knng;
        }
    }

    fn get_queries(&self, start_idx: usize, end_idx: usize) -> Vec<(usize, u8)> { // block_idx, type
        let mut queries = Vec::new();
        queries.reserve(20);

        let mut now_idx = self.root_idx;

        loop {
            let block = &self.blocks[now_idx];
            if start_idx <= block.start_idx && block.end_idx <= end_idx && block.dummy == false {
                queries.push((now_idx, if block.complete {KNN} else {BF}));
                if !has_left_sibling(now_idx, block.level) {break;}
                now_idx = get_left_sibling(now_idx, block.level);
            } else if end_idx < block.start_idx || block.end_idx < start_idx {
                if !has_left_sibling(now_idx, block.level) {break;}
                now_idx = get_left_sibling(now_idx, block.level);
            } else if self.threshold <= ((min(block.end_idx, end_idx) - max(block.start_idx, start_idx) + 1) as f32) / ((block.end_idx - block.start_idx + 1) as f32) && block.dummy == false {
                queries.push((now_idx, if block.complete {KNN} else {BF}));
                if !has_left_sibling(now_idx, block.level) {break;}
                now_idx = get_left_sibling(now_idx, block.level);
            } else {
                if block.level == 0 && block.dummy == false {
                    queries.push((now_idx, BF));
                }
                if now_idx == 0 {break;}
                now_idx = now_idx - 1;
            }
        }

        return queries;
    }

    pub fn new(tknn_leaf_size: usize, n_neighbors: usize, max_candidates: usize, max_iter: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str, k: usize) -> TknnMBI {
        let blocks = Vec::new();
        let data = Array2::zeros((0, 0));
        let label = Array1::zeros(0);
        let is_normed = match dist {
            "normed_cosine" => true,
            "normed_squared_cosine" => true,
            _ => false
        };

        TknnMBI {
            blocks,
            root_idx: 0,
            growing_idx: 0,
            data,
            label,
            tknn_leaf_size,
            n_neighbors,
            max_candidates,
            max_iter,
            rpdiv_iter,
            rpdiv_size_limit,
            tf_eps,
            dist: String::from(dist),
            k,
            is_normed,
            bf_limit: tknn_leaf_size,
            time_get_query: 0.0,
            time_query: 0.0,
            time_refine_result: 0.0,
            threshold: 0.7,
        }
    }

    fn binary_search(&self, start_label: i64, end_label: i64) -> (usize, usize) {
        let mut lo = -1;
        let mut hi = self.data.shape()[0] as i32 - 1;
        while lo+1 < hi {
            let mid = ((lo + hi) / 2) as usize;
            if start_label <= self.label[mid] {hi = mid as i32}
            else {lo = mid as i32}
        }
        let start_idx = hi as usize;

        let mut lo = 0;
        let mut hi = self.data.shape()[0] as i32;
        while lo+1 < hi {
            let mid = ((lo + hi) / 2) as usize;
            if end_label < self.label[mid] {hi = mid as i32}
            else {lo = mid as i32}
        }
        let end_idx = lo as usize;
        return (start_idx, end_idx);
    }

    // result를 병합하는 과정에서 최적화의 여지가 있지만, 암달의 법칙때매 보류
    pub fn single_query(&mut self, q: &[f32], start_label: i64, end_label: i64) -> (Vec<i32>, Vec<f32>, usize){
        let data = self.data.as_slice().unwrap();
        let dim = self.data.shape()[1];
        let mut cnt = 0;
        let dist = match self.dist.as_str() {
            "normed_cosine" => normed_cosine_distance,
            "normed_squared_cosine" => normed_squared_cosine_distance,
            _ => euclidean_distance
        };

        let t1 = Instant::now();
        let (start_idx, end_idx) = self.binary_search(start_label, end_label);

        if (self.bf_limit != 0) && (end_idx-start_idx <= self.bf_limit){
            // println!("bf {} {}", start_idx, end_idx);
            let mut res_indices = vec![-1 ; self.k];
            let mut res_dists = vec![f32::INFINITY ; self.k];
            for idx in start_idx..(end_idx+1) {
                heap_push(idx as i32, dist(q, vec_of(idx, data, dim)), &mut res_indices, &mut res_dists);
            }
            return (res_indices, res_dists, cnt);
        }
        // println!("notbf {} {}", start_idx, end_idx);

        let queries = self.get_queries(start_idx, end_idx);
        // println!("queries: {:?}", queries);
        let t2 = Instant::now();

        let mut result: Vec<(i32, f32)> = Vec::new(); // (idx, dist), keep k nearest neighbors
        result.reserve(self.k);
        let mut counts: Vec<usize> = Vec::new(); // counts of compare operations, each block
        counts.reserve(queries.len());
        for (block_idx, query_type) in queries {

            let mut now;
            if query_type == KNN { //&& level != 0 {
                let knng = &mut self.blocks[block_idx].knng;
                let (indices, dists, _cnt) = knng.single_query(q, start_label, end_label);
                let k = self.k;
                cnt += _cnt;
                now = (0..min(k, indices.len())).map(|i| (self.blocks[block_idx].start_idx as i32 + indices[i], dists[i])).collect::<Vec<(i32, f32)>>();
                now.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                // counts.push(_cnt);
            } else {
                now = Vec::new();
                let mut _cnt = 0;
                let dim = self.data.shape()[1];
                let mut res_indices = vec![-1 ; self.k];
                let mut res_dists = vec![f32::INFINITY ; self.k];
                for idx in max(self.blocks[block_idx].start_idx, start_idx)..min(self.blocks[block_idx].end_idx, end_idx)+1 {
                    heap_push(idx as i32, dist(q, vec_of(idx, data, dim)), &mut res_indices, &mut res_dists);
                    _cnt += 1;
                }
                // heap to vec
                for i in 0..self.k {
                    if res_indices[i] == -1 {break;}
                    now.push((res_indices[i], res_dists[i]));
                }
                now.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                // counts.push(_cnt);
                cnt += _cnt;
            }
            
            result = sorted_merge(&result, &now, self.k);
        }
        // println!("counts: {:?}", counts);
        // println!("cnt: {:?}", cnt);
        let t3 = Instant::now();

        // return result;
        let mut indices = Vec::new();
        let mut dists = Vec::new();
        indices.reserve(self.k);
        dists.reserve(self.k);
        for (idx, dist) in result {
            indices.push(idx);
            dists.push(dist);
        }
        let t4 = Instant::now();
        self.time_get_query += (t2-t1).as_secs_f64();
        self.time_query += (t3-t2).as_secs_f64();
        self.time_refine_result += (t4-t3).as_secs_f64();
        return (indices, dists, cnt);
    }
}

#[pymethods]
impl TknnMBI {
    #[staticmethod]
    pub fn dump<'py>(knng: &TknnMBI, path: &str){
        let mut file = File::create(path).unwrap();
        let encoded = bincode::serialize(knng).unwrap();
        file.write_all(&encoded).unwrap();
    }

    #[staticmethod]
    pub fn load<'py>(path: &str) -> Self{
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        let decoded: TknnMBI = bincode::deserialize(&buffer[..]).unwrap();
        return decoded;
    }

    #[staticmethod]
    pub fn build_graph<'py>(data: PyReadonlyArray2<f32>, label: PyReadonlyArray1<i64>, n_neighbors: usize, max_candidates: usize,
        max_iters: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str, k: usize, tknn_leaf_size: usize) -> Self {
        
        let s_time = time::SystemTime::now();
        let mut tknn = TknnMBI::new(tknn_leaf_size, n_neighbors, max_candidates, max_iters, rpdiv_iter, rpdiv_size_limit, tf_eps, dist, k);

        let data_array = data.as_array().to_owned();
        // let (n_data, dim) = (data.shape()[0], data.shape()[1]);
        let label_array = label.as_array().to_owned();
        tknn.add_data(data_array, label_array);
        
        println!("TknnMBI::build_graph: {:?}", s_time.elapsed());
        
        return tknn;
    }

    #[staticmethod]
    pub fn build_graph_parallel<'py>(data: PyReadonlyArray2<f32>, label: PyReadonlyArray1<i64>, n_neighbors: usize, max_candidates: usize,
        max_iters: usize, rpdiv_iter: usize, rpdiv_size_limit: usize, tf_eps: f32, dist: &str, k: usize, tknn_leaf_size: usize) -> Self {
        
        let s_time = time::SystemTime::now();
        let mut tknn = TknnMBI::new(tknn_leaf_size, n_neighbors, max_candidates, max_iters, rpdiv_iter, rpdiv_size_limit, tf_eps, dist, k);

        let data_array = data.as_array().to_owned();
        // let (n_data, dim) = (data.shape()[0], data.shape()[1]);
        let label_array = label.as_array().to_owned();
        tknn.add_data_parallel(data_array, label_array);
        
        println!("TknnMBI::build_graph: {:?}", s_time.elapsed());
        
        return tknn;
    }

    #[pyo3(name="single_query")]
    fn py_single_query<'py>(&mut self, py: Python<'py>, q: PyReadonlyArray1<f32>, start_label: i64, end_label: i64) -> (&'py PyArray1<i32>, &'py PyArray1<f32>, usize) {
        let (res_indices, res_dists, cnt) =
            if self.is_normed {
                let q_slice = q.as_slice().unwrap();
                let q_denom = q_slice.iter().map(|x| *x * *x).sum::<f32>().sqrt();
                let q_vec: Vec<f32> = q_slice.iter().map(|x| *x/q_denom).collect::<Vec<f32>>();

                self.single_query(&q_vec, start_label, end_label)
            } else {
                self.single_query(q.as_slice().unwrap(), start_label, end_label)
            };
        
        return (
            PyArray1::from_vec(py, res_indices),
            PyArray1::from_vec(py, res_dists),
            cnt
        );
    }

    pub fn set_query_params(&mut self, k: usize, eps:f32, leaf_size: usize, threshold: f32, bf_limit: usize) {
        self.k = k;
        for block in self.blocks.iter_mut() {
            if block.complete {block.knng.set_query_params(k, eps, leaf_size);}
        }
        self.threshold = threshold;
        self.bf_limit = bf_limit;
    }

    pub fn get_time(&self) -> (f64, f64, f64) {
        return (self.time_get_query, self.time_query, self.time_refine_result);
    }

    pub fn clear_time(&mut self) {
        self.time_get_query = 0.0;
        self.time_query = 0.0;
        self.time_refine_result = 0.0;
    }

    #[pyo3(name="add_data")]
    pub fn py_add_data<'py>(&mut self, data: PyReadonlyArray2<f32>, label: PyReadonlyArray1<i64>) {
        let data_array = data.as_array().to_owned();
        let label_array = label.as_array().to_owned();
        self.add_data(data_array, label_array);
    }
    
    #[pyo3(name="add_data_parallel")]
    pub fn py_add_data_parallel<'py>(&mut self, data: PyReadonlyArray2<f32>, label: PyReadonlyArray1<i64>) {
        let data_array = data.as_array().to_owned();
        let label_array = label.as_array().to_owned();
        self.add_data_parallel(data_array, label_array);
    }
}