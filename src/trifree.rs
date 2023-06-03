use std::{cmp::Ordering};
use rustc_hash::FxHashMap;
// use crate::{knng::KNNGraph, nndescent::nndescent, distances::euclidean_distance_avx};


struct Edge { u: i32, v: i32, d: f32 }

#[allow(dead_code)]
pub struct CSR {pub indptr: Vec<i32>, pub indices: Vec<i32>, elems: Vec<f32>}

pub fn make_trifree(neighbors: &[i32], distances: &[f32], k: usize, eps: f32) -> CSR{
    
    let edges = get_sorted_unique_edges(neighbors, distances, k);

    let n_nodes = neighbors.len() / k;
    let mut adj: Vec<FxHashMap<i32, f32>> = vec![FxHashMap::default(); n_nodes];
    // type SimpleHasher = BuildHasherDefault<simple::SimpleHasher>;
    // let mut adj = Vec::with_capacity(n_nodes);
    // for _ in 0..n_nodes {
    //     adj.push(HashMap::<i32, f32, SimpleHasher>::default());
    // }

    for e in edges {
        
        if !has_bypass(e.u, e.v, eps, &adj) {
            adj[e.u as usize].insert(e.v, e.d);
        }

        if !has_bypass(e.v, e.u, eps, &adj) {
            adj[e.v as usize].insert(e.u, e.d);
        }
    }
    
    let mut indptr = vec![0i32; n_nodes + 1];
    
    for u in 0..n_nodes {
        indptr[u+1] = indptr[u] + adj[u].len() as i32;
    }

    let sz = indptr[indptr.len()-1] as usize;
    let mut indices = Vec::with_capacity(sz);
    let mut elems = Vec::with_capacity(sz);

    for u in 0..n_nodes {
        let mut nn = Vec::from_iter(adj[u].iter());
        nn.sort_by(|a, b| a.1.total_cmp(b.1));
        for (v, d) in nn {
            indices.push(*v);
            elems.push(*d);
        }
    }

    return CSR {indptr, indices, elems};
    
}

fn has_bypass(s: i32, t: i32, eps: f32, adj: &Vec<FxHashMap<i32, f32>>) -> bool {
    for (m, smd) in adj[s as usize].iter() {
        match adj[*m as usize].get(&t) {
            Some(mtd) => {
                if *smd > *mtd * eps {
                    return true;
                }
            },
            None => {/*do nothing*/}
        }
    }
    return false;
}

fn get_sorted_unique_edges(neighbors: &[i32], distances: &[f32], k: usize) -> Vec<Edge> {
    let n_nodes = neighbors.len() / k;
    let mut idx = 0;
    let mut edges: Vec<Edge>  = vec![];
    for u in 0..n_nodes as i32 {
        if u == -1 { continue; }
        for _ in 0..k {

            let v = neighbors[idx];
            if v == -1 { continue; }
            let d = distances[idx];

            if u < v {
                edges.push(Edge {u, v, d});
            }
            else {
                edges.push(Edge {u: v, v: u, d});
            }

            idx += 1;
        }
    }
    edges.sort_by(|a, b| {
        match a.u.cmp(&b.u) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.v.cmp(&b.v)
        }
    });
    edges.dedup_by(|a, b| a.u == b.u && a.v == b.v);
    edges.sort_by(|a, b| a.d.total_cmp(&b.d));
    edges
}

pub fn remove_long_edges_eps(indptr: &[i32], indices: &[i32], elems: &[f32], eps: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>){

    const PLAIN: i8 = 0;
    const REMOVED: i8 = 1;
    const RESERVED: i8 = 2;

    let mut flags = vec![PLAIN; indices.len()]; // 0: normal, 1: removed, 2: reserved
    
    for u in 0..indptr.len()-1 {

        for i in (indptr[u] as usize)..(indptr[u+1] as usize) {

            let v = indices[i] as usize;
            
            let (mut ucur, uend) = (indptr[u] as usize, indptr[u+1] as usize);
            let (mut vcur, vend) = (indptr[v] as usize, indptr[v+1] as usize);
            
            while ucur < uend && vcur < vend {
                if indices[ucur] < indices[vcur] {
                    ucur += 1;
                }
                else if indices[ucur] > indices[vcur] {
                    vcur += 1;
                }
                else {
                    
                    let mut edges = [i, ucur, vcur];
                    edges.sort_by(|a, b| elems[*a].total_cmp(&elems[*b]));
                    
                    // if all three edges are not removed
                    if (flags[edges[0]] != REMOVED) && (flags[edges[1]] != REMOVED) && (flags[edges[2]] != REMOVED) {
                        // if the longest edge is not reserved and satisfies the eps condition
                        if flags[edges[2]] != RESERVED && (elems[edges[0]] + elems[edges[1]]) < elems[edges[2]] * eps {
                            // remove the longest edge
                            flags[edges[2]] = REMOVED;
                            // reserve remaining edges
                            flags[edges[0]] = RESERVED;
                            flags[edges[1]] = RESERVED;
                        }
                    }

                    ucur += 1;
                    vcur += 1;
                }
            }

        }
      
    }

    let n_edges = flags.iter().filter(|x| **x != REMOVED).count();

    let mut new_indptr = vec![0; indptr.len()];
    let mut new_indices = vec![0; n_edges];
    let mut new_elems = vec![0f32; n_edges];
    
    let mut nnz = 0usize;
    let mut i = 0usize;
    for u in 0..new_indptr.len()-1 {
        while i < indptr[u+1] as usize {
            if flags[i] != REMOVED {
                new_indices[nnz] = indices[i];
                new_elems[nnz] = elems[i];
                nnz += 1;
            }
            i += 1;
        }
        new_indptr[u+1] = nnz as i32;
    }

    return (new_indptr, new_indices, new_elems);

}

pub fn make_bidirectional_graph(indptr: &[i32], indices: &[i32], elems: &[f32]) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut new_indptr = vec![0; indptr.len()];
    let mut new_indices = vec![0; indices.len() * 2];
    let mut new_elems = vec![0f32; elems.len() * 2];

    for u in 0..indptr.len()-1 {
        new_indptr[u] += indptr[u+1] - indptr[u];
        for i in indptr[u]..indptr[u+1] {
            let v = indices[i as usize] as usize;
            new_indptr[v] += 1;
        }
    }
    
    for u in 0..new_indptr.len()-1 {
        new_indptr[u+1] += new_indptr[u];
    }

    for u in 0..indptr.len()-1 {
        let cpsz = (indptr[u+1] - indptr[u]) as usize;
        
        let slice_indices = &mut new_indices[new_indptr[u] as usize-cpsz..new_indptr[u] as usize];
        slice_indices.copy_from_slice(&indices[indptr[u] as usize..indptr[u+1] as usize]);

        let slice_elems = &mut new_elems[new_indptr[u] as usize-cpsz..new_indptr[u] as usize];
        slice_elems.copy_from_slice(&elems[indptr[u] as usize..indptr[u+1] as usize]);

        new_indptr[u] -= cpsz as i32;
    }
    
    for u in 0..indptr.len()-1 {
        for i in indptr[u]..indptr[u+1] {
            let v = indices[i as usize] as usize;
            let vd = elems[i as usize];

            new_indptr[v] -= 1;
            new_indices[new_indptr[v] as usize] = u as i32;
            new_elems[new_indptr[v] as usize] = vd;
        }
    }

    for u in 0..new_indptr.len()-1 {
        let u_indices = &mut new_indices[new_indptr[u] as usize .. new_indptr[u+1] as usize];
        let u_elems = &mut new_elems[new_indptr[u] as usize .. new_indptr[u+1] as usize];
        let mut ids: Vec<usize> = (0..u_indices.len()).collect();
        ids.sort_by(|a, b| u_elems[*a].total_cmp(&u_elems[*b]));
        
        let x: Vec<i32> = ids.iter().map(|x| u_indices[*x]).collect();
        for (a, b) in u_indices.iter_mut().zip(x) { *a = b; }

        let x: Vec<f32> = ids.iter().map(|x| u_elems[*x]).collect();
        for (a, b) in u_elems.iter_mut().zip(x) { *a = b; }
    }

    return (new_indptr, new_indices, new_elems);

}


pub fn make_oriented_graph(neighbors: &[i32], distances: &[f32], k: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>){
    let n_data = neighbors.len() / k;
    let mut deg = vec![k; n_data];
    
    for u_neighbors in neighbors.chunks(k) {
        for v in u_neighbors {
            deg[*v as usize] += 1;
        }
    }

    let mut indptr = vec![0i32; n_data+1];
    let mut indices = vec![0i32; k*n_data];
    let mut elems = vec![0.0f32; k*n_data];

    for (u, u_neighbors) in neighbors.chunks(k).enumerate() {
        for v in u_neighbors {
            let v = *v as usize;
            if deg[u] < deg[v] || (deg[u] == deg[v] && u < v) {
                indptr[u] += 1;
            }
            else{
                indptr[v] += 1;
            }
        }
    }
    
    for i in 0..n_data {
        indptr[i+1] += indptr[i];
    }

    let nchunks = neighbors.chunks(k);
    let dchunks = distances.chunks(k);
    for (u, (u_neighbors, u_distances)) in nchunks.zip(dchunks).enumerate() {
        for (v, d) in u_neighbors.iter().zip(u_distances) {
            let v = *v as usize;
            if deg[u] < deg[v] || (deg[u] == deg[v] && u < v) {
                indptr[u] -= 1;
                indices[indptr[u] as usize] = v as i32;
                elems[indptr[u] as usize] = *d;
            }
            else {
                indptr[v] -= 1;
                indices[indptr[v] as usize] = u as i32;
                elems[indptr[v] as usize] = *d;
            }
        }
    }

    let mut tmp = vec![(0i32, 0f32); 0];

    // sort by id
    for u in 0..n_data {

        tmp.truncate(0);

        for i in indptr[u]..indptr[u+1] {
            tmp.push((indices[i as usize], elems[i as usize]));
        }
        
        tmp.sort_unstable_by_key(|x| x.0);

        for (i, (idx, elem)) in (indptr[u]..indptr[u+1]).zip(tmp.iter()) {
            indices[i as usize] = *idx;
            elems[i as usize] = *elem;
        }
    }

    // dedup
    let mut nnz = 0usize;
    let mut row_end = 0usize;
    for u in 0..n_data {
        let mut i = row_end;
        row_end = indptr[u+1] as usize;
        while i < row_end {
            let idx = indices[i];
            let elem = elems[i];
            i += 1;
            while i < row_end && idx == indices[i] {
                // println!("{:?} {:?}", i, row_end);
                i += 1;
            }
            indices[nnz] = idx;
            elems[nnz] = elem;
            nnz += 1;
        }
        indptr[u+1] = nnz as i32;
    }

    indices.truncate(nnz);
    elems.truncate(nnz);
    

    return (indptr, indices, elems);

}


// #[test]
// fn make_trifree_test() {
//     use fastrand;
//     use crate::{nndescent::nndescent, distances::euclidean_distance_avx};
    
//     let mut data = vec![];
//     let dim = 2;
    
//     for _ in 0..100 {
//         data.push(fastrand::f32());
//     }
    
//     let (neighbors, distances) = nndescent(&data, dim, 10, 10, 0, 10, 100, euclidean_distance_avx);
//     let neighbors = neighbors.as_slice().unwrap();
//     let distances = distances.as_slice().unwrap();

//     let csr = make_trifree(neighbors, distances, 10, 1.0);

//     let n_data = data.len() / dim;
//     for u in 0..n_data {
//         println!("{:?}", &csr.indices[csr.indptr[u] as usize..csr.indptr[u+1] as usize]);
//         println!("{:?}", &csr.elems[csr.indptr[u] as usize..csr.indptr[u+1] as usize]);
//     }

    
    

// }
