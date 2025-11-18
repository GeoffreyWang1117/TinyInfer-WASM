---
name: Performance Issue
about: Report performance problems or suggest optimizations
title: '[PERF] '
labels: performance
assignees: ''
---

## ⚡ Performance Issue

<!-- Describe the performance problem you're experiencing -->

## 📊 Benchmark Results

<!-- Provide benchmark numbers showing the issue -->

**Current Performance:**
- Operation:
- Time:
- Throughput:

**Expected Performance:**
- Operation:
- Time:
- Throughput:

## 🔍 Profiling Data

<!-- If you've done profiling, share the results -->

```
Paste profiling output or flamegraph link here
```

## 🌍 Environment

- **Hardware**: [e.g. M1 MacBook Pro, Intel i7-10700K]
- **Browser**: [e.g. Chrome 119]
- **OS**: [e.g. macOS 13.0]
- **SIMD Enabled**: [Yes/No]
- **Workers Used**: [Yes/No]

## 📝 Steps to Reproduce

1. Load model '...'
2. Run inference with input '...'
3. Measure time using '...'

## 💡 Suggested Optimization

<!-- If you have ideas for how to improve performance, share them -->

## 📈 Impact

<!-- How much would fixing this improve your use case? -->

- [ ] Critical - Makes the library unusable
- [ ] High - Significantly impacts user experience
- [ ] Medium - Noticeable but manageable
- [ ] Low - Minor improvement

## 📋 Additional Context

<!-- Any other context about the performance issue -->

## ✅ Checklist

- [ ] I have measured performance using the built-in profiler
- [ ] I have compared with baseline benchmarks
- [ ] I have tested with SIMD enabled
- [ ] I have tested with Workers enabled
