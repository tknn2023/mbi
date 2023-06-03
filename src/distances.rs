#[cfg(target_arch="x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch="x86_64")]
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut aiter = a.chunks_exact(8);
        let mut biter = b.chunks_exact(8);

        let mut acc = _mm256_setzero_ps();

        aiter.by_ref().zip(biter.by_ref()).for_each(|(ac, bc)| {
            acc = acc_mul_256(&ac[0], &bc[0], acc);
        });
        
        let arem = aiter.remainder();
        let brem = biter.remainder();
        
        let acc_sum = hsum256_ps_avx(acc) +
                arem.iter().zip(brem).map(|(x, y)| {
                    x * y
                }).sum::<f32>();

        return acc_sum;
    }
}

#[cfg(target_arch="x86_64")]
#[inline(always)]
unsafe fn acc_mul_256(ac: &f32, bc: &f32, acc: __m256) -> __m256 {
    let av = _mm256_loadu_ps(ac);
    let bv = _mm256_loadu_ps(bc);
    return _mm256_add_ps(_mm256_mul_ps(av, bv), acc);
}

#[cfg(target_arch="x86_64")]
#[inline(always)]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32{
    unsafe {
        
        let mut aiter = a.chunks_exact(8);
        let mut biter = b.chunks_exact(8);
        
        let mut acc = _mm256_setzero_ps();

        aiter.by_ref().zip(biter.by_ref()).for_each(|(ac, bc)| {
            acc = acc_diff_square_256(&ac[0], &bc[0], acc);
        });

        let arem = aiter.remainder();
        let brem = biter.remainder();
        
        let acc_sum = hsum256_ps_avx(acc) +
                arem.iter().zip(brem).map(|(x, y)| {
                    let diff = *x - *y;
                    diff * diff
                }).sum::<f32>();

        return acc_sum.sqrt();
    }
}

#[cfg(target_arch="x86_64")]
#[inline(always)]
unsafe fn acc_diff_square_256(ac: &f32, bc: &f32, acc: __m256) -> __m256 {
    let av = _mm256_loadu_ps(ac);
    let bv = _mm256_loadu_ps(bc);
    let diff = _mm256_sub_ps(av, bv);
    return _mm256_add_ps(_mm256_mul_ps(diff, diff), acc);
}

#[cfg(target_arch="x86_64")]
#[inline(always)]
fn hsum256_ps_avx(v: __m256) -> f32{
    unsafe {
        let x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
        let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }
}



#[cfg(not(target_arch="x86_64"))]
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(ax,bx)| {
        ax * bx
    }).sum()
}

#[cfg(not(target_arch="x86_64"))]
#[inline(always)]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32{
    let y: f32 = a.iter().zip(b).map(|(ax,bx)| {
        let x = *ax - *bx;
        x*x
    }).sum();
    y.sqrt()
}

#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let aa = dot(a, a);
    let bb = dot(b, b);
    let ab = dot(a, b);
    let aabb = aa * bb;

    if aabb > 0f32 {
        return 1f32 - 1f32 * ab / aabb.sqrt();
    }
    else {
        return 1.0;
    }
}


#[inline(always)]
pub fn normed_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    return 1f32 - dot(a, b);
}

#[inline(always)]
pub fn normed_squared_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    return (1f32 - dot(a, b)).max(0.0).sqrt();
}


