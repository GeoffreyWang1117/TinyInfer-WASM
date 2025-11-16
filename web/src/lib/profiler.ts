/**
 * 性能分析工具
 * 提供详细的推理性能分析和可视化
 */

export interface PerformanceMetrics {
  totalTime: number
  initTime: number
  inferenceTime: number
  postProcessTime: number
  memoryUsed: number
  throughput: number
  layerTimings: Map<string, number>
}

export interface LayerProfile {
  name: string
  type: string
  inputShape: number[]
  outputShape: number[]
  execTime: number
  percentage: number
}

export class PerformanceProfiler {
  private startTime: number = 0
  private metrics: PerformanceMetrics
  private layerTimings: Map<string, number[]> = new Map()
  private memoryBefore: number = 0

  constructor() {
    this.metrics = {
      totalTime: 0,
      initTime: 0,
      inferenceTime: 0,
      postProcessTime: 0,
      memoryUsed: 0,
      throughput: 0,
      layerTimings: new Map(),
    }
  }

  /**
   * 开始性能分析
   */
  startProfiling() {
    this.startTime = performance.now()
    this.memoryBefore = this.getMemoryUsage()
  }

  /**
   * 记录层执行时间
   */
  recordLayer(layerName: string, execTime: number) {
    if (!this.layerTimings.has(layerName)) {
      this.layerTimings.set(layerName, [])
    }
    this.layerTimings.get(layerName)!.push(execTime)
  }

  /**
   * 结束性能分析
   */
  endProfiling(): PerformanceMetrics {
    const totalTime = performance.now() - this.startTime
    const memoryAfter = this.getMemoryUsage()

    // 计算平均层时间
    const avgLayerTimings = new Map<string, number>()
    this.layerTimings.forEach((times, layer) => {
      const avg = times.reduce((a, b) => a + b, 0) / times.length
      avgLayerTimings.set(layer, avg)
    })

    this.metrics = {
      totalTime,
      initTime: 0, // 可以细化
      inferenceTime: totalTime,
      postProcessTime: 0,
      memoryUsed: memoryAfter - this.memoryBefore,
      throughput: 1000 / totalTime, // inferences per second
      layerTimings: avgLayerTimings,
    }

    return this.metrics
  }

  /**
   * 获取内存使用量（MB）
   */
  private getMemoryUsage(): number {
    if ('memory' in performance && (performance as any).memory) {
      return (performance as any).memory.usedJSHeapSize / 1024 / 1024
    }
    return 0
  }

  /**
   * 生成性能报告
   */
  generateReport(): string {
    const { totalTime, inferenceTime, memoryUsed, throughput, layerTimings } = this.metrics

    let report = '📊 性能分析报告\n'
    report += '=' .repeat(50) + '\n\n'

    report += `⏱️  总时间: ${totalTime.toFixed(2)}ms\n`
    report += `🔄 推理时间: ${inferenceTime.toFixed(2)}ms\n`
    report += `💾 内存使用: ${memoryUsed.toFixed(2)}MB\n`
    report += `🚀 吞吐量: ${throughput.toFixed(2)} inferences/sec\n\n`

    if (layerTimings.size > 0) {
      report += '📈 层级性能:\n'
      report += '-'.repeat(50) + '\n'

      const sortedLayers = Array.from(layerTimings.entries())
        .sort((a, b) => b[1] - a[1])

      sortedLayers.forEach(([layer, time]) => {
        const percentage = (time / inferenceTime) * 100
        const bar = '█'.repeat(Math.floor(percentage / 2))
        report += `${layer.padEnd(20)} ${time.toFixed(2)}ms ${bar} ${percentage.toFixed(1)}%\n`
      })
    }

    return report
  }

  /**
   * 获取层级性能分析
   */
  getLayerProfiles(): LayerProfile[] {
    const { inferenceTime, layerTimings } = this.metrics
    const profiles: LayerProfile[] = []

    layerTimings.forEach((time, name) => {
      profiles.push({
        name,
        type: this.inferLayerType(name),
        inputShape: [],  // 需要从实际推理中获取
        outputShape: [],
        execTime: time,
        percentage: (time / inferenceTime) * 100,
      })
    })

    return profiles.sort((a, b) => b.execTime - a.execTime)
  }

  /**
   * 推断层类型
   */
  private inferLayerType(name: string): string {
    if (name.includes('conv')) return 'Convolution'
    if (name.includes('matmul') || name.includes('gemm')) return 'MatMul'
    if (name.includes('relu')) return 'Activation'
    if (name.includes('pool')) return 'Pooling'
    if (name.includes('norm')) return 'Normalization'
    return 'Unknown'
  }

  /**
   * 获取性能指标
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics }
  }

  /**
   * 重置分析器
   */
  reset() {
    this.layerTimings.clear()
    this.startTime = 0
    this.memoryBefore = 0
  }
}

/**
 * 批量性能测试
 */
export class BatchProfiler {
  private profiler: PerformanceProfiler
  private results: PerformanceMetrics[] = []

  constructor() {
    this.profiler = new PerformanceProfiler()
  }

  /**
   * 运行批量测试
   */
  async runBatch(
    inferenceFunc: () => Promise<void> | void,
    iterations: number = 10
  ): Promise<PerformanceMetrics[]> {
    this.results = []

    for (let i = 0; i < iterations; i++) {
      this.profiler.reset()
      this.profiler.startProfiling()

      await inferenceFunc()

      const metrics = this.profiler.endProfiling()
      this.results.push(metrics)
    }

    return this.results
  }

  /**
   * 获取统计信息
   */
  getStatistics() {
    if (this.results.length === 0) {
      throw new Error('No results available. Run batch test first.')
    }

    const times = this.results.map(r => r.totalTime)
    const sorted = times.sort((a, b) => a - b)

    return {
      mean: times.reduce((a, b) => a + b, 0) / times.length,
      median: sorted[Math.floor(sorted.length / 2)],
      min: Math.min(...times),
      max: Math.max(...times),
      stdDev: this.calculateStdDev(times),
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p90: sorted[Math.floor(sorted.length * 0.9)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
    }
  }

  /**
   * 计算标准差
   */
  private calculateStdDev(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    return Math.sqrt(variance)
  }

  /**
   * 生成统计报告
   */
  generateStatisticsReport(): string {
    const stats = this.getStatistics()

    let report = '📊 批量测试统计\n'
    report += '='.repeat(50) + '\n\n'
    report += `📈 样本数: ${this.results.length}\n`
    report += `⚡ 平均值: ${stats.mean.toFixed(2)}ms\n`
    report += `📊 中位数: ${stats.median.toFixed(2)}ms\n`
    report += `⬇️  最小值: ${stats.min.toFixed(2)}ms\n`
    report += `⬆️  最大值: ${stats.max.toFixed(2)}ms\n`
    report += `📏 标准差: ${stats.stdDev.toFixed(2)}ms\n\n`
    report += '📊 百分位数:\n'
    report += `  P50: ${stats.p50.toFixed(2)}ms\n`
    report += `  P90: ${stats.p90.toFixed(2)}ms\n`
    report += `  P95: ${stats.p95.toFixed(2)}ms\n`
    report += `  P99: ${stats.p99.toFixed(2)}ms\n`

    return report
  }
}

/**
 * 性能比较工具
 */
export class PerformanceComparator {
  private results: Map<string, PerformanceMetrics[]> = new Map()

  /**
   * 添加测试结果
   */
  addResult(name: string, metrics: PerformanceMetrics) {
    if (!this.results.has(name)) {
      this.results.set(name, [])
    }
    this.results.get(name)!.push(metrics)
  }

  /**
   * 比较结果
   */
  compare(): Map<string, any> {
    const comparison = new Map<string, any>()

    this.results.forEach((metrics, name) => {
      const avgTime = metrics.reduce((sum, m) => sum + m.totalTime, 0) / metrics.length
      const avgThroughput = metrics.reduce((sum, m) => sum + m.throughput, 0) / metrics.length

      comparison.set(name, {
        avgTime,
        avgThroughput,
        samples: metrics.length,
      })
    })

    return comparison
  }

  /**
   * 生成对比报告
   */
  generateComparisonReport(): string {
    const comparison = this.compare()
    const entries = Array.from(comparison.entries())

    if (entries.length === 0) {
      return '暂无对比数据'
    }

    // 找到最快的实现
    const fastest = entries.reduce((a, b) =>
      a[1].avgTime < b[1].avgTime ? a : b
    )

    let report = '⚔️  性能对比\n'
    report += '='.repeat(50) + '\n\n'

    entries
      .sort((a, b) => a[1].avgTime - b[1].avgTime)
      .forEach(([name, stats]) => {
        const speedup = fastest[1].avgTime / stats.avgTime
        const isFastest = name === fastest[0]

        report += `${isFastest ? '🥇' : '  '} ${name}\n`
        report += `   时间: ${stats.avgTime.toFixed(2)}ms\n`
        report += `   吞吐: ${stats.avgThroughput.toFixed(2)} ops/sec\n`
        report += `   加速: ${speedup.toFixed(2)}x\n\n`
      })

    return report
  }
}

/**
 * 性能建议系统
 */
export class PerformanceAdvisor {
  /**
   * 分析性能并给出建议
   */
  analyze(metrics: PerformanceMetrics): string[] {
    const suggestions: string[] = []

    // 检查总体性能
    if (metrics.totalTime > 100) {
      suggestions.push('⚠️ 推理时间较长 (>100ms)，考虑使用更小的模型或优化算子')
    }

    // 检查内存使用
    if (metrics.memoryUsed > 100) {
      suggestions.push('💾 内存使用较高 (>100MB)，考虑使用内存池或减少中间张量')
    }

    // 检查层级性能
    if (metrics.layerTimings.size > 0) {
      const layerArray = Array.from(metrics.layerTimings.entries())
      const sortedLayers = layerArray.sort((a, b) => b[1] - a[1])
      const slowestLayer = sortedLayers[0]

      if (slowestLayer[1] / metrics.inferenceTime > 0.5) {
        suggestions.push(`🐌 ${slowestLayer[0]} 占用了 ${((slowestLayer[1] / metrics.inferenceTime) * 100).toFixed(1)}% 的时间，建议优化此层`)
      }
    }

    // 检查吞吐量
    if (metrics.throughput < 10) {
      suggestions.push('🚀 吞吐量较低 (<10 ops/sec)，考虑使用 SIMD 加速或算子融合')
    }

    if (suggestions.length === 0) {
      suggestions.push('✅ 性能表现良好！')
    }

    return suggestions
  }

  /**
   * 生成优化建议报告
   */
  generateReport(metrics: PerformanceMetrics): string {
    const suggestions = this.analyze(metrics)

    let report = '💡 性能优化建议\n'
    report += '='.repeat(50) + '\n\n'

    suggestions.forEach((suggestion, index) => {
      report += `${index + 1}. ${suggestion}\n`
    })

    return report
  }
}

// 导出便捷函数
export function createProfiler(): PerformanceProfiler {
  return new PerformanceProfiler()
}

export function createBatchProfiler(): BatchProfiler {
  return new BatchProfiler()
}

export function createComparator(): PerformanceComparator {
  return new PerformanceComparator()
}

export function createAdvisor(): PerformanceAdvisor {
  return new PerformanceAdvisor()
}
