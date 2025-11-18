import { test, expect } from '@playwright/test'

/**
 * Worker Demo E2E Tests
 */

test.describe('Worker Demo Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/worker')
  })

  test('should load worker demo page', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('Web Worker')
    await expect(page.locator('text=UI 响应性测试')).toBeVisible()
  })

  test('should show UI counter running', async ({ page }) => {
    // Get initial counter value
    const counter = page.locator('div.text-5xl')
    const initial = await counter.textContent()

    // Wait a bit
    await page.waitForTimeout(500)

    // Counter should have increased
    const current = await counter.textContent()
    expect(parseInt(current || '0')).toBeGreaterThan(parseInt(initial || '0'))
  })

  test('should initialize worker', async ({ page }) => {
    // Click initialize button
    await page.click('button:has-text("Initialize Worker")')

    // Wait for worker to be ready
    await expect(page.locator('text=Worker initialized')).toBeVisible({ timeout: 3000 })
    await expect(page.locator('text=Ready')).toBeVisible()
  })

  test('should run single inference', async ({ page }) => {
    // Initialize worker first
    await page.click('button:has-text("Initialize Worker")')
    await expect(page.locator('text=Ready')).toBeVisible({ timeout: 3000 })

    // Run single inference
    await page.click('button:has-text("Run Single Inference")')

    // Wait for results
    await expect(page.locator('text=Inference Results')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('table')).toBeVisible()
  })

  test('should run parallel inferences', async ({ page }) => {
    // Initialize worker first
    await page.click('button:has-text("Initialize Worker")')
    await expect(page.locator('text=Ready')).toBeVisible({ timeout: 3000 })

    // Set parallel count to 3
    await page.locator('input[type="range"]').fill('3')

    // Run parallel inferences
    await page.click('button:has-text("Run")')

    // Wait for results
    await expect(page.locator('text=Inference Results')).toBeVisible({ timeout: 10000 })

    // Should have 3 results
    const rows = page.locator('tbody tr')
    await expect(rows).toHaveCount(3)
  })

  test('UI should remain responsive during inference', async ({ page }) => {
    // Initialize worker
    await page.click('button:has-text("Initialize Worker")')
    await expect(page.locator('text=Ready')).toBeVisible({ timeout: 3000 })

    // Get counter before inference
    const counter = page.locator('div.text-5xl')
    const before = await counter.textContent()

    // Start inference
    await page.click('button:has-text("Run Single Inference")')

    // Counter should still be updating (UI not blocked)
    await page.waitForTimeout(100)
    const during = await counter.textContent()

    expect(parseInt(during || '0')).toBeGreaterThan(parseInt(before || '0'))
  })

  test('should terminate worker', async ({ page }) => {
    // Initialize worker
    await page.click('button:has-text("Initialize Worker")')
    await expect(page.locator('text=Ready')).toBeVisible({ timeout: 3000 })

    // Terminate
    await page.click('button:has-text("Terminate Worker")')

    // Status should show terminated
    await expect(page.locator('text=Worker terminated')).toBeVisible()
  })
})
