/// Memory pool management for efficient tensor allocation
/// Reduces allocation overhead by reusing buffers

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory pool for reusing tensor buffers
/// Uses a simple size-based pooling strategy
pub struct MemoryPool {
    /// Pools organized by buffer size (in elements)
    pools: HashMap<usize, Vec<Vec<f32>>>,
    /// Maximum number of buffers to keep per size
    max_buffers_per_size: usize,
    /// Statistics
    stats: PoolStats,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_allocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub total_freed: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_buffers_per_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_buffers_per_size,
            stats: PoolStats::default(),
        }
    }

    /// Create a new memory pool with default settings
    pub fn default() -> Self {
        Self::new(8) // Keep up to 8 buffers per size
    }

    /// Allocate a buffer of the specified size
    /// Tries to reuse from pool first, allocates new if needed
    pub fn allocate(&mut self, size: usize) -> Vec<f32> {
        self.stats.total_allocations += 1;

        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                self.stats.pool_hits += 1;
                return buffer;
            }
        }

        self.stats.pool_misses += 1;
        vec![0.0; size]
    }

    /// Return a buffer to the pool for reuse
    pub fn free(&mut self, mut buffer: Vec<f32>) {
        self.stats.total_freed += 1;

        let size = buffer.len();
        let pool = self.pools.entry(size).or_insert_with(Vec::new);

        // Only keep buffer if pool isn't full
        if pool.len() < self.max_buffers_per_size {
            // Clear buffer before returning to pool
            buffer.fill(0.0);
            pool.push(buffer);
        }
        // Otherwise, let it drop and be freed
    }

    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Get hit rate (percentage of allocations served from pool)
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_allocations == 0 {
            0.0
        } else {
            (self.stats.pool_hits as f64 / self.stats.total_allocations as f64) * 100.0
        }
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        self.stats = PoolStats::default();
    }

    /// Get total number of buffers in all pools
    pub fn total_pooled_buffers(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }

    /// Get memory usage estimate (in bytes)
    pub fn memory_usage(&self) -> usize {
        self.pools
            .iter()
            .map(|(size, buffers)| size * buffers.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Thread-safe memory pool
pub struct SharedMemoryPool {
    pool: Arc<Mutex<MemoryPool>>,
}

impl SharedMemoryPool {
    pub fn new(max_buffers_per_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(MemoryPool::new(max_buffers_per_size))),
        }
    }

    pub fn default() -> Self {
        Self::new(8)
    }

    pub fn allocate(&self, size: usize) -> Vec<f32> {
        self.pool.lock().unwrap().allocate(size)
    }

    pub fn free(&self, buffer: Vec<f32>) {
        self.pool.lock().unwrap().free(buffer)
    }

    pub fn stats(&self) -> PoolStats {
        self.pool.lock().unwrap().stats().clone()
    }

    pub fn hit_rate(&self) -> f64 {
        self.pool.lock().unwrap().hit_rate()
    }

    pub fn clear(&self) {
        self.pool.lock().unwrap().clear()
    }
}

impl Clone for SharedMemoryPool {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
        }
    }
}

/// RAII-style buffer handle that automatically returns to pool
pub struct PooledBuffer {
    buffer: Option<Vec<f32>>,
    pool: SharedMemoryPool,
}

impl PooledBuffer {
    pub fn new(size: usize, pool: SharedMemoryPool) -> Self {
        let buffer = pool.allocate(size);
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get reference to buffer data
    pub fn data(&self) -> &[f32] {
        self.buffer.as_ref().unwrap()
    }

    /// Get mutable reference to buffer data
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.buffer.as_mut().unwrap()
    }

    /// Take ownership of buffer (prevents auto-return to pool)
    pub fn take(mut self) -> Vec<f32> {
        self.buffer.take().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.free(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = MemoryPool::new(4);

        // First allocation should miss
        let buf1 = pool.allocate(100);
        assert_eq!(buf1.len(), 100);
        assert_eq!(pool.stats().pool_misses, 1);
        assert_eq!(pool.stats().pool_hits, 0);

        // Return to pool
        pool.free(buf1);
        assert_eq!(pool.stats().total_freed, 1);

        // Second allocation should hit
        let buf2 = pool.allocate(100);
        assert_eq!(buf2.len(), 100);
        assert_eq!(pool.stats().pool_hits, 1);

        pool.free(buf2);
    }

    #[test]
    fn test_memory_pool_multiple_sizes() {
        let mut pool = MemoryPool::new(4);

        // Allocate different sizes
        let buf1 = pool.allocate(100);
        let buf2 = pool.allocate(200);
        let buf3 = pool.allocate(100);

        assert_eq!(pool.stats().total_allocations, 3);
        assert_eq!(pool.stats().pool_misses, 3); // All misses

        pool.free(buf1);
        pool.free(buf2);
        pool.free(buf3);

        // Now try to reuse
        let buf4 = pool.allocate(100); // Hit
        let buf5 = pool.allocate(200); // Hit
        let buf6 = pool.allocate(300); // Miss

        assert_eq!(pool.stats().pool_hits, 2);
        assert_eq!(pool.stats().pool_misses, 4);

        pool.free(buf4);
        pool.free(buf5);
        pool.free(buf6);
    }

    #[test]
    fn test_pool_size_limit() {
        let mut pool = MemoryPool::new(2); // Max 2 buffers per size

        // Allocate and free 3 buffers of same size
        for _ in 0..3 {
            let buf = pool.allocate(100);
            pool.free(buf);
        }

        // Pool should only keep 2 buffers
        assert_eq!(pool.total_pooled_buffers(), 2);
    }

    #[test]
    fn test_hit_rate() {
        let mut pool = MemoryPool::new(4);

        // 5 allocations, first is miss, rest hit same buffer
        let buf = pool.allocate(100);
        pool.free(buf);

        for _ in 0..4 {
            let buf = pool.allocate(100);
            pool.free(buf);
        }

        // Hit rate should be 80% (4 hits out of 5 total)
        assert!((pool.hit_rate() - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_pooled_buffer_raii() {
        let pool = SharedMemoryPool::new(4);

        {
            let mut buf = PooledBuffer::new(100, pool.clone());
            buf.data_mut()[0] = 42.0;
            // Buffer automatically returned to pool when dropped
        }

        // Next allocation should reuse the buffer
        assert_eq!(pool.stats().pool_hits, 1);
    }

    #[test]
    fn test_shared_pool() {
        let pool = SharedMemoryPool::new(4);

        let buf1 = pool.allocate(100);
        pool.free(buf1);

        let buf2 = pool.allocate(100);
        assert_eq!(pool.stats().pool_hits, 1);

        pool.free(buf2);
    }
}
