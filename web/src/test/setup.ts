/**
 * Vitest Setup File
 *
 * This file runs before all tests and sets up the testing environment.
 */

import { expect, afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// Cleanup after each test
afterEach(() => {
  cleanup()
})

// Extend Vitest matchers if needed
// expect.extend(customMatchers)

// Mock WebAssembly if not available in test environment
if (typeof WebAssembly === 'undefined') {
  global.WebAssembly = {
    instantiate: async () => ({
      instance: {
        exports: {},
      },
      module: {},
    }),
    compile: async () => ({}),
    validate: () => true,
    Module: class {},
    Instance: class {},
    Memory: class {},
    Table: class {},
    CompileError: Error,
    LinkError: Error,
    RuntimeError: Error,
  } as any
}

// Mock Performance API if needed
if (typeof performance === 'undefined') {
  global.performance = {
    now: () => Date.now(),
  } as any
}

console.log('✓ Vitest setup complete')
