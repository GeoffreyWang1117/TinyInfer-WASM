/**
 * IndexedDB Model Cache for TinyInfer
 *
 * Caches parsed models in browser storage to avoid re-parsing on subsequent loads.
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb'
import type { ModelDef } from './onnxLoader'

/**
 * Database schema
 */
interface ModelCacheDB extends DBSchema {
  models: {
    key: string // Model URL or file hash
    value: {
      modelDef: ModelDef
      timestamp: number
      metadata: {
        name: string
        size: number
        format: 'onnx' | 'json'
      }
    }
  }
}

/**
 * Model Cache Manager
 */
export class ModelCache {
  private db: IDBPDatabase<ModelCacheDB> | null = null
  private readonly DB_NAME = 'tinyinfer-models'
  private readonly DB_VERSION = 1
  private readonly STORE_NAME = 'models'

  /**
   * Initialize the IndexedDB database
   */
  async initialize(): Promise<void> {
    if (this.db) return

    this.db = await openDB<ModelCacheDB>(this.DB_NAME, this.DB_VERSION, {
      upgrade(db) {
        // Create models store if it doesn't exist
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'key' })
        }
      },
    })
  }

  /**
   * Get cached model by key
   */
  async get(key: string): Promise<ModelDef | null> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    const cached = await this.db.get(this.STORE_NAME, key)
    if (!cached) {
      return null
    }

    console.log(`[ModelCache] Cache hit for: ${key}`)
    return cached.modelDef
  }

  /**
   * Store model in cache
   */
  async set(
    key: string,
    modelDef: ModelDef,
    metadata: { name: string; size: number; format: 'onnx' | 'json' }
  ): Promise<void> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    await this.db.put(this.STORE_NAME, {
      key,
      modelDef,
      timestamp: Date.now(),
      metadata,
    })

    console.log(`[ModelCache] Cached model: ${key}`)
  }

  /**
   * Check if model exists in cache
   */
  async has(key: string): Promise<boolean> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    const cached = await this.db.get(this.STORE_NAME, key)
    return cached !== undefined
  }

  /**
   * Delete cached model
   */
  async delete(key: string): Promise<void> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    await this.db.delete(this.STORE_NAME, key)
    console.log(`[ModelCache] Deleted cached model: ${key}`)
  }

  /**
   * Clear all cached models
   */
  async clear(): Promise<void> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    await this.db.clear(this.STORE_NAME)
    console.log('[ModelCache] Cleared all cached models')
  }

  /**
   * Get all cached model keys
   */
  async getAllKeys(): Promise<string[]> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    return await this.db.getAllKeys(this.STORE_NAME)
  }

  /**
   * Get cache statistics
   */
  async getStats(): Promise<{
    count: number
    totalSize: number
    models: Array<{ key: string; name: string; size: number; timestamp: number }>
  }> {
    await this.initialize()

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    const all = await this.db.getAll(this.STORE_NAME)
    const totalSize = all.reduce((sum, item) => sum + item.metadata.size, 0)

    return {
      count: all.length,
      totalSize,
      models: all.map((item) => ({
        key: item.key,
        name: item.metadata.name,
        size: item.metadata.size,
        timestamp: item.timestamp,
      })),
    }
  }

  /**
   * Generate cache key from File
   */
  static async generateKeyFromFile(file: File): Promise<string> {
    // Use file name, size, and last modified as key
    const keyData = `${file.name}-${file.size}-${file.lastModified}`
    const encoder = new TextEncoder()
    const data = encoder.encode(keyData)
    const hashBuffer = await crypto.subtle.digest('SHA-256', data)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')
    return `file-${hashHex}`
  }

  /**
   * Generate cache key from URL
   */
  static generateKeyFromURL(url: string): string {
    return `url-${url}`
  }
}

/**
 * Create a singleton instance
 */
const modelCache = new ModelCache()

/**
 * Get cached model
 */
export async function getCachedModel(key: string): Promise<ModelDef | null> {
  return modelCache.get(key)
}

/**
 * Cache model
 */
export async function cacheModel(
  key: string,
  modelDef: ModelDef,
  metadata: { name: string; size: number; format: 'onnx' | 'json' }
): Promise<void> {
  return modelCache.set(key, modelDef, metadata)
}

/**
 * Check if model is cached
 */
export async function hasCachedModel(key: string): Promise<boolean> {
  return modelCache.has(key)
}

/**
 * Delete cached model
 */
export async function deleteCachedModel(key: string): Promise<void> {
  return modelCache.delete(key)
}

/**
 * Clear all cached models
 */
export async function clearModelCache(): Promise<void> {
  return modelCache.clear()
}

/**
 * Get cache statistics
 */
export async function getModelCacheStats() {
  return modelCache.getStats()
}

/**
 * Generate cache key from file
 */
export async function generateCacheKeyFromFile(file: File): Promise<string> {
  return ModelCache.generateKeyFromFile(file)
}

/**
 * Generate cache key from URL
 */
export function generateCacheKeyFromURL(url: string): string {
  return ModelCache.generateKeyFromURL(url)
}

export default modelCache
