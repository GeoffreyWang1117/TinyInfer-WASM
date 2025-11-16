/**
 * 性能基准测试示例
 * 演示如何使用 Benchmark API 进行性能测试
 */

import { initWasm, createBenchmark } from '../web/src/lib/tinyinfer'

async function benchmarkExample() {
  console.log('=== 性能基准测试示例 ===\n')

  // 初始化
  await initWasm()
  const bench = createBenchmark()

  console.log('运行性能测试...\n')

  // 1. MatMul 基准测试
  console.log('1️⃣ MatMul (512x512) 基准测试')
  const matmulTime = bench.benchmarkMatMul(512, 10)
  console.log(`   时间: ${matmulTime.toFixed(2)}ms`)
  console.log(`   吞吐量: ${(10 / (matmulTime / 1000)).toFixed(1)} iterations/s\n`)

  // 2. ReLU 基准测试
  console.log('2️⃣ ReLU (100K elements) 基准测试')
  const reluTime = bench.benchmarkReLU(100000, 100)
  console.log(`   时间: ${reluTime.toFixed(2)}ms`)
  console.log(`   吞吐量: ${(100 / (reluTime / 1000)).toFixed(1)} iterations/s`)
  console.log(`   元素处理速度: ${((100000 * 100) / (reluTime / 1000) / 1e6).toFixed(1)}M elements/s\n`)

  // 3. Conv2D 基准测试
  console.log('3️⃣ Conv2D (1x64x224x224) 基准测试')
  const convTime = bench.benchmarkConv2D(1, 64, 224, 5)
  console.log(`   时间: ${convTime.toFixed(2)}ms`)
  console.log(`   吞吐量: ${(5 / (convTime / 1000)).toFixed(2)} iterations/s\n`)

  // 4. 获取所有结果
  const results = bench.getResults()
  console.log('所有测试结果:', Array.from(results))

  // 5. 性能分析
  console.log('\n📊 性能分析:')
  console.log(`   最快: ${Math.min(...Array.from(results)).toFixed(2)}ms`)
  console.log(`   最慢: ${Math.max(...Array.from(results)).toFixed(2)}ms`)
  console.log(`   平均: ${(Array.from(results).reduce((a, b) => a + b, 0) / results.length).toFixed(2)}ms`)

  // 清理
  bench.free()
  console.log('\n✅ 基准测试完成')
}

benchmarkExample().catch(console.error)
