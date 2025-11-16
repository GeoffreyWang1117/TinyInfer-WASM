/// SIMD optimization utilities for WebAssembly
/// Provides SIMD-accelerated operations using wasm32 SIMD intrinsics

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// Check if SIMD is available at runtime
#[inline]
pub fn is_simd_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        cfg!(target_feature = "simd128")
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// SIMD-accelerated vector addition: c[i] = a[i] + b[i]
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i + 4 <= len {
        unsafe {
            // Load 4 elements from each array
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            // Add vectors
            let vc = f32x4_add(va, vb);

            // Store result
            v128_store(c.as_mut_ptr().add(i) as *mut v128, vc);
        }
        i += 4;
    }

    // Handle remaining elements
    for j in i..len {
        c[j] = a[j] + b[j];
    }
}

/// SIMD-accelerated vector multiplication: c[i] = a[i] * b[i]
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_mul_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = f32x4_mul(va, vb);
            v128_store(c.as_mut_ptr().add(i) as *mut v128, vc);
        }
        i += 4;
    }

    for j in i..len {
        c[j] = a[j] * b[j];
    }
}

/// SIMD-accelerated vector subtraction: c[i] = a[i] - b[i]
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_sub_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = f32x4_sub(va, vb);
            v128_store(c.as_mut_ptr().add(i) as *mut v128, vc);
        }
        i += 4;
    }

    for j in i..len {
        c[j] = a[j] - b[j];
    }
}

/// SIMD-accelerated vector division: c[i] = a[i] / b[i]
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_div_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = f32x4_div(va, vb);
            v128_store(c.as_mut_ptr().add(i) as *mut v128, vc);
        }
        i += 4;
    }

    for j in i..len {
        c[j] = a[j] / b[j];
    }
}

/// SIMD-accelerated ReLU: c[i] = max(0, a[i])
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_relu_f32(a: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    unsafe {
        let zero = f32x4_splat(0.0);

        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vc = f32x4_max(va, zero);
            v128_store(c.as_mut_ptr().add(i) as *mut v128, vc);
            i += 4;
        }
    }

    for j in i..len {
        c[j] = a[j].max(0.0);
    }
}

/// SIMD-accelerated dot product: sum(a[i] * b[i])
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut i = 0;

    unsafe {
        let mut sum = f32x4_splat(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vprod = f32x4_mul(va, vb);
            sum = f32x4_add(sum, vprod);
            i += 4;
        }

        // Extract and sum the 4 elements
        let sum_array: [f32; 4] = std::mem::transmute(sum);
        let mut result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for j in i..len {
            result += a[j] * b[j];
        }

        result
    }
}

/// SIMD-accelerated FMA: c[i] = a[i] * b[i] + c[i]
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_fma_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let len = a.len();
    let mut i = 0;

    while i + 4 <= len {
        unsafe {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = v128_load(c.as_ptr().add(i) as *const v128);

            let vprod = f32x4_mul(va, vb);
            let vresult = f32x4_add(vprod, vc);

            v128_store(c.as_mut_ptr().add(i) as *mut v128, vresult);
        }
        i += 4;
    }

    for j in i..len {
        c[j] += a[j] * b[j];
    }
}

/// SIMD-accelerated scalar multiplication: b[i] = a[i] * scalar
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn simd_scale_f32(a: &[f32], scalar: f32, b: &mut [f32]) {
    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut i = 0;

    unsafe {
        let vscalar = f32x4_splat(scalar);

        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = f32x4_mul(va, vscalar);
            v128_store(b.as_mut_ptr().add(i) as *mut v128, vb);
            i += 4;
        }
    }

    for j in i..len {
        b[j] = a[j] * scalar;
    }
}

// Fallback implementations for non-WASM targets
#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_mul_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_sub_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] - b[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_div_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] / b[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_relu_f32(a: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i].max(0.0);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_fma_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] += a[i] * b[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn simd_scale_f32(a: &[f32], scalar: f32, b: &mut [f32]) {
    for i in 0..a.len() {
        b[i] = a[i] * scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 5];

        simd_add_f32(&a, &b, &mut c);

        assert_eq!(c, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_simd_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut c = vec![0.0; 4];

        simd_mul_f32(&a, &b, &mut c);

        assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_relu() {
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut c = vec![0.0; 5];

        simd_relu_f32(&a, &mut c);

        assert_eq!(c, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_simd_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = simd_dot_f32(&a, &b);

        assert_eq!(result, 10.0);
    }
}
