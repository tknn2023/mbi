use fastrand;

#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone)]
pub struct VPTree {
    pub nodes: Vec<i32>,
    pub sizes: Vec<usize>,
    pub radii: Vec<f32>
}

#[derive(Debug)]
struct Item {
    nid: i32,
    dist: f32
}

#[inline(always)]
fn vec_of(v :usize, data: &[f32], dim: usize) -> &[f32] {
    return &data[v*dim..v*dim+dim];
}

impl VPTree {
    pub fn new(data: &[f32], dim: usize, dist: fn(&[f32], &[f32]) -> f32) -> Self {
        let n_data = data.len() / dim;
        let mut vptree = VPTree {
            nodes: (0..n_data as i32).collect(),
            sizes: vec![1; n_data],
            radii: vec![0f32; n_data]
        };

        vptree.construct_rec(0, n_data, data, dim, dist);

        return vptree;
    }
    
    fn construct_rec(&mut self, l: usize, r: usize, data: &[f32], dim: usize, dist: fn(&[f32], &[f32]) -> f32) {
        
        if r - l <= 1 {
            return;
        }

        self.sizes[l] = r - l;

        let pivot_idx = fastrand::usize(l..r);
        let pivot = self.nodes[pivot_idx];
        self.nodes[pivot_idx] = self.nodes[l];
        self.nodes[l] = pivot; // swap
        
        let mut lst = Vec::<Item>::with_capacity(r-l-1);
        for nid in &self.nodes[l+1..r] {
            lst.push(
                Item {
                    nid: *nid,
                    dist: dist(vec_of(pivot as usize, data, dim), vec_of(*nid as usize, data, dim))
                }
            );
        }

        lst.sort_by(|a,b| a.dist.total_cmp(&b.dist));

        for (v, b) in self.nodes[l+1..r].iter_mut().zip(&lst) {
            *v = b.nid;
        }
        
        let mid = (l+1+r) / 2;
        
        self.radii[l] = if mid >= l+2 {
            lst[mid - l - 1].dist + lst[mid - l - 2].dist
        }
        else {
            lst[mid - l - 1].dist
        } / 2.0;

        self.construct_rec(l+1, mid, data, dim, dist);
        self.construct_rec(mid, r, data, dim, dist);


    }
}

#[test]
fn vptree_test() {
    use crate::distances::euclidean_distance;

    let data = [0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

    let dim = 1;
    
    let vptree = VPTree::new(&data, dim, euclidean_distance);
    
    println!("{:?}", data);
    println!("{:?}", vptree.nodes);
    println!("{:?}", vptree.sizes);
    println!("{:?}", vptree.radii);
}