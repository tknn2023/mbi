#[inline]
pub fn heap_push(v: i32, d: f32, data: &mut[i32], priorities: &mut[f32]) -> usize{

    if priorities[0] <= d {
        return 0;
    }

    data[0] = v as i32;
    priorities[0] = d;

    siftdown(0, data, priorities);
    return 1;
}

#[inline]
pub fn heap_push_no_dupl(v: usize, d: f32, data: &mut[i32], priorities: &mut[f32]) -> usize{

    if priorities[0] <= d {
        return 0;
    }

    let v = v as i32;

    for w in data.iter() {
        if *w == v {
            return 0;
        }
    }

    data[0] = v;
    priorities[0] = d;

    siftdown(0, data, priorities);
    return 1;
}

#[inline]
fn siftdown(mut pos: usize, data: &mut[i32], priorities: &mut[f32]) {
    let size = data.len();
    
    let me = data[pos];
    let me_dist = priorities[pos];
    
    let mut child = pos*2+1;

    while child < size {
        let right = child + 1;

        if right < size && priorities[child] < priorities[right] {
            child += 1;
        }

        if me_dist >= priorities[child] {
            break;
        }

        data[pos] = data[child];
        priorities[pos] = priorities[child];
        pos = child;
        child = pos*2+1;
    }

    data[pos] = me;
    priorities[pos] = me_dist;
}


#[inline]
pub fn heap_push_no_dupl_flagged(v: usize, d: f32, data: &mut [i32], priorities: &mut [f32], flags: &mut [bool]) -> usize{

    if priorities[0] <= d {
        return 0;
    }

    let v = v as i32;

    for w in data.iter() {
        if *w == v {
            return 0;
        }
    }

    data[0] = v;
    priorities[0] = d;
    flags[0] = true; // The neighbor added here always has a 'new' flag.

    siftdown_flagged(0, data, priorities, flags);
    return 1;
}



#[inline]
fn siftdown_flagged(mut pos: usize, neighbors_of_u: &mut [i32], distances_to_u: &mut [f32], flags_of_u: &mut [bool]) {
    let size = neighbors_of_u.len();
    
    let me = neighbors_of_u[pos];
    let me_dist = distances_to_u[pos];
    let me_flag = flags_of_u[pos];
    
    let mut child = pos*2+1;

    while child < size {
        let right = child + 1;

        if right < size && distances_to_u[child] < distances_to_u[right] {
            child += 1;
        }

        if me_dist >= distances_to_u[child] {
            break;
        }

        neighbors_of_u[pos] = neighbors_of_u[child];
        distances_to_u[pos] = distances_to_u[child];
        flags_of_u[pos] = flags_of_u[child]; 
        pos = child;
        child = pos*2+1;
    }

    neighbors_of_u[pos] = me;
    distances_to_u[pos] = me_dist;
    flags_of_u[pos] = me_flag; 
}



#[inline]
pub fn cand_pop(cand_indices: &mut Vec<i32>, cand_dists: &mut Vec<f32>) -> (i32, f32) {

    let sz = cand_indices.len()-1;
    let ret = (cand_indices[0], cand_dists[0]);
    cand_indices[0] = cand_indices[sz];
    cand_dists[0] = cand_dists[sz];
    cand_indices.truncate(sz);
    cand_dists.truncate(sz);

    pushdown(0, cand_indices, cand_dists);

    return ret;
}

#[inline]
pub fn cand_push(v: i32, vd: f32, cand_indices: &mut Vec<i32>, cand_dists: &mut Vec<f32>) {

    cand_indices.push(v);
    cand_dists.push(vd);

    pushup(cand_indices.len()-1, cand_indices, cand_dists);
}

#[inline]
fn pushup(mut me: usize, cand_indices: &mut [i32], cand_dists: &mut [f32]) {
    let my_idx = cand_indices[me];
    let my_dist = cand_dists[me];

    while me > 0 {
        let next = (me - 1) / 2;
        let next_dist = cand_dists[next];
    
        if  my_dist >= next_dist {
            break;
        }

        cand_indices[me] = cand_indices[next];
        cand_dists[me] = next_dist;

        me = next;
    }
    cand_indices[me] = my_idx;
    cand_dists[me] = my_dist;
}

#[inline]
fn pushdown(mut me: usize, cand_indices: &mut [i32], cand_dists: &mut [f32]) {
    let size = cand_indices.len();
    
    if size == 0 { return; }

    let me_idx = cand_indices[me] as usize;
    let me_dist = cand_dists[me];
    
    loop {
        let mut child = me*2+1;

        if child >= size { break; }
        
        let right = child + 1;

        if right < size && cand_dists[right] < cand_dists[child] {
            child += 1;
        }

        if cand_dists[child] >= me_dist { break; }

        cand_indices[me] = cand_indices[child];
        cand_dists[me] = cand_dists[child];
        me = child;
    }

    cand_indices[me] = me_idx as i32;
    cand_dists[me] = me_dist;
}


#[test]
fn cand_pop_push_test() {
    let mut cand_indices = vec![0; 0];
    let mut cand_dists = vec![0f32; 0];


    cand_push(4, 4.0, &mut cand_indices, &mut cand_dists);
    cand_push(5, 5.0, &mut cand_indices, &mut cand_dists);
    cand_push(6, 6.0, &mut cand_indices, &mut cand_dists);
    cand_push(7, 7.0, &mut cand_indices, &mut cand_dists);
    cand_push(2, 2.0, &mut cand_indices, &mut cand_dists);
    cand_push(1, 1.0, &mut cand_indices, &mut cand_dists);
    cand_push(9, 9.0, &mut cand_indices, &mut cand_dists);
    cand_push(3, 3.0, &mut cand_indices, &mut cand_dists);

    println!("cand_indices: {:?}", cand_indices);
    println!("cand_dists: {:?}", cand_dists);

    let ret = cand_pop(&mut cand_indices, &mut cand_dists);
    
    println!("{:?}", ret);

    println!("cand_indices: {:?}", cand_indices);
    println!("cand_dists: {:?}", cand_dists);

    let ret = cand_pop(&mut cand_indices, &mut cand_dists);
    
    println!("{:?}", ret);

    println!("cand_indices: {:?}", cand_indices);
    println!("cand_dists: {:?}", cand_dists);

    let ret = cand_pop(&mut cand_indices, &mut cand_dists);
    
    println!("{:?}", ret);

    println!("cand_indices: {:?}", cand_indices);
    println!("cand_dists: {:?}", cand_dists);

}