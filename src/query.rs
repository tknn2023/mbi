use crate::heap::heap_push;

// const VISITED: u8 = 2;
// const QUEUED: u8 = 1;
const VISITED: u8 = 1;
const UNVISITED: u8 = 0;

// struct SearchQuery<'a> {
//     q : &'a [f32],
//     k : usize,
//     max_candidates: usize,
//     search_width: usize,
//     eps: f32
// }

// struct SearchMaterials {
//     cand_indices : Vec<i32>,
//     cand_dists: Vec<f32>,
//     visited: Vec<u8>
// }

// static mut SM : SearchMaterials = SearchMaterials {
//     cand_indices: vec![],
//     cand_dists: vec![],
//     visited: vec![]
// };

pub fn single_query(q: &[f32], init_cands: &[i32], init_dists: &[f32], k: usize, max_candidates: usize, width: usize, eps: f32,
        indptr: &[i32], indices: &[i32], data: &[f32], dim: usize,
        visited: &mut [u8], dist: fn(&[f32], &[f32]) -> f32) -> (Vec<i32>, Vec<f32>, usize) {

    visited.fill(0);

    // TEST CODE
    // let x = unsafe{&mut SM.cand_indices};
    // x.push(3);

    let mut res_indices = vec![-1 ; k];
    let mut res_dists = vec![f32::INFINITY ; k];
    let mut cand_indices: Vec<i32> = Vec::with_capacity(max_candidates);
    let mut cand_dists: Vec<f32> = Vec::with_capacity(max_candidates);
    
    let mut cnt = 0;

    //TODO: heapify
    for (v, vd) in init_cands.iter().zip(init_dists) {
        cnt += 1;
        cand_push(*v, *vd, &mut cand_indices, &mut cand_dists, max_candidates);
        visited[*v as usize] = VISITED;
    }
    
    while cand_indices.len() > 0 {

        let (u, d) = cand_pop(&mut cand_indices, &mut cand_dists);

        if d >= res_dists[0] * eps {
            break;
        }

        heap_push(u, d, &mut res_indices, &mut res_dists);

        for ind in indptr[u as usize] as usize..(indptr[u as usize +1] as usize).min(indptr[u as usize] as usize + width) {
            let v = indices[ind] as usize;
            
            if visited[v] == UNVISITED {
                let vd = dist(q, &data[v*dim..v*dim+dim]);
                cnt += 1;
                if vd < res_dists[0] * eps{
                    cand_push(v as i32, vd, &mut cand_indices, &mut cand_dists, max_candidates);
                }
            }
            visited[v] = VISITED;
        }
    }

    return (res_indices, res_dists, cnt);
}

#[inline]
fn cand_push(v: i32, vd: f32, cand_indices: &mut Vec<i32>, cand_dists: &mut Vec<f32>, max_candidates: usize) {
    if cand_indices.len() < max_candidates {
        cand_indices.push(v);
        cand_dists.push(vd);
        pushup(cand_indices.len()-1, cand_indices, cand_dists);
    }
    else {
        let maxid = if cand_dists[1] > cand_dists[2] {1} else {2};

        cand_dists[maxid] = vd;
        cand_indices[maxid] = v;
    
        if cand_dists[maxid] < cand_dists[0] {
            cand_dists.swap(0, maxid);
            cand_indices.swap(0, maxid);
        }
    
        pushdown_max(maxid, cand_indices, cand_dists);
    }
}

#[inline]
fn cand_pop(indices: &mut Vec<i32>, dists: &mut Vec<f32>) -> (i32, f32){

    let ret = (indices[0], dists[0]);

    if indices.len() > 1 {
        (indices[0], dists[0]) = (indices.pop().unwrap(), dists.pop().unwrap());
    
        pushdown_min(0, indices, dists);
    }
    else{
        indices.pop();
        dists.pop();
    }

    return ret;
}

fn pushdown_min(mut me: usize, data: &mut [i32], priorities: &mut [f32]) {
    
    let size = data.len();

    let mut child = me * 2 + 1;

    while child < size {
        let right = child + 1;
        if right < size {
            if priorities[right] < priorities[child] {
                child = right;
            }
            let lmgc = me * 4 + 3; //left most grand child
            if lmgc < size {
                let rmgc = (me * 4 + 7).min(size); //right most grand child (exclusive)
                for gc in lmgc..rmgc {
                    if priorities[gc] < priorities[child] {
                        child = gc;
                    }
                }
            }
        }

        if priorities[child] < priorities[me] {
            data.swap(child, me);
            priorities.swap(child, me);
            if child > me * 2 + 2 { // if `child` is a grandchild
                let parent_of_child = (child-1)/2;
                if priorities[child] > priorities[parent_of_child] {
                    data.swap(child, parent_of_child);
                    priorities.swap(child, parent_of_child);
                }
            }
        }
        else{
            break;
        }

        me = child;
        child = me * 2 + 1;
    }
}

fn pushdown_max(mut me: usize, data: &mut [i32], priorities: &mut [f32]) {
    
    let size = data.len();

    let mut child = me * 2 + 1;

    while child < size {
        let right = child + 1;
        if right < size {
            if priorities[right] > priorities[child] {
                child = right;
            }
            let lmgc = me * 4 + 3; //left most grand child
            if lmgc < size {
                let rmgc = (me * 4 + 7).min(size); //right most grand child (exclusive)
                for gc in lmgc..rmgc {
                    if priorities[gc] > priorities[child] {
                        child = gc;
                    }
                }
            }
        }

        if priorities[child] > priorities[me] {
            data.swap(child, me);
            priorities.swap(child, me);
            if child > me * 2 + 2 { // if `child` is a grandchild
                let parent_of_child = (child-1)/2;
                if priorities[child] < priorities[parent_of_child] {
                    data.swap(child, parent_of_child);
                    priorities.swap(child, parent_of_child);
                }
            }
        }
        else{
            break;
        }

        me = child;
        child = me * 2 + 1;
    }
}

#[inline(always)]
pub fn is_min_level(i: usize) -> bool {
    (i+1).leading_zeros() % 2 == 1
}


fn pushup(me: usize, data: &mut [i32], priorities: &mut [f32]) {
    if me > 0 {
        let parent = (me - 1) / 2;
        if is_min_level(me) {
            if priorities[me] > priorities[parent] {
                data.swap(me, parent);
                priorities.swap(me, parent);
                pushup_max(parent, data, priorities);
            }
            else{
                pushup_min(me, data, priorities);
            }
        }
        else {
            if priorities[me] < priorities[parent] {
                data.swap(me, parent);
                priorities.swap(me, parent);
                pushup_min(parent, data, priorities);
            }
            else{
                pushup_max(me, data, priorities);
            }
        }
    }
}

fn pushup_min(mut me: usize, data: &mut [i32], priorities: &mut [f32]) {
    //me > 2: `me` has a grand parent
    while me > 2 && priorities[me] < priorities[((me + 1) >> 2) - 1]{ 
        
        data.swap(me, ((me + 1) >> 2) - 1);
        priorities.swap(me, ((me + 1) >> 2) - 1);

        me = ((me + 1) >> 2) - 1;
    }
}

fn pushup_max(mut me: usize, data: &mut [i32], priorities: &mut [f32]) {
    //me > 2: `me` has a grand parent
    while me > 2 && priorities[me] > priorities[((me + 1) >> 2) - 1]{ 
        
        data.swap(me, ((me + 1) >> 2) - 1);
        priorities.swap(me, ((me + 1) >> 2) - 1);

        me = ((me + 1) >> 2) - 1;
    }
}


pub fn tree_search_one(q: &[f32], data: &[f32], dim: usize, left: &[i32], right: &[i32], dist: fn(&[f32], &[f32]) -> f32) -> (i32, f32, usize) {
    
    let mut me = 0usize;
    let mut min_id = 0;
    let mut min_dist = dist(q, &data[0..dim]);
    
    let mut cnt = 0;

    loop {
        let mut u = -1i32;
        let mut d = f32::INFINITY;
        if left[me] != -1 {
            u = left[me];
            d = dist(q, &data[(u as usize)*dim..(u as usize)*dim+dim]);
            cnt += 1;
        }
        if right[me] != -1 {
            let v = right[me] as usize;
            let vd = dist(q, &data[v*dim..v*dim+dim]);
            cnt += 1;
            if vd < d {
                u = v as i32;
                d = vd;
            }
        }

        if u != -1 {
            me = u as usize;
            if d < min_dist {
                min_id = u;
                min_dist = d;
            }
        }
        else {
            break;
        }
    }

    return (min_id as i32, min_dist, cnt);

}
