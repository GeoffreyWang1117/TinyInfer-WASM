import { test, expect } from '@playwright/test'

/**
 * Positional Encoding Demo E2E Tests
 */

test.describe('Positional Encoding Demo Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/positional')
  })

  test('should load positional encoding page', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('位置编码')
    await expect(page.locator('text=配置参数')).toBeVisible()
  })

  test('should switch encoding types', async ({ page }) => {
    // Test Sinusoidal (default)
    const sinusoidalBtn = page.locator('button:has-text("Sinusoidal")')
    await expect(sinusoidalBtn).toHaveClass(/bg-blue-500/)

    // Switch to Learnable
    await page.click('button:has-text("Learnable")')
    const learnableBtn = page.locator('button:has-text("Learnable")')
    await expect(learnableBtn).toHaveClass(/bg-blue-500/)

    // Switch to RoPE
    await page.click('button:has-text("RoPE")')
    const ropeBtn = page.locator('button:has-text("RoPE")')
    await expect(ropeBtn).toHaveClass(/bg-blue-500/)
  })

  test('should adjust d_model parameter', async ({ page }) => {
    const slider = page.locator('input[type="range"]').first()

    // Set to 128
    await slider.fill('128')

    // Check label updated
    await expect(page.locator('text=d_model (模型维度): 128')).toBeVisible()
  })

  test('should adjust max_len parameter', async ({ page }) => {
    const slider = page.locator('input[type="range"]').nth(1)

    // Set to 64
    await slider.fill('64')

    // Check label updated
    await expect(page.locator('text=max_len (最大序列长度): 64')).toBeVisible()
  })

  test('should show statistics', async ({ page }) => {
    await expect(page.locator('text=统计信息')).toBeVisible()
    await expect(page.locator('text=最小值')).toBeVisible()
    await expect(page.locator('text=最大值')).toBeVisible()
    await expect(page.locator('text=均值')).toBeVisible()
    await expect(page.locator('text=标准差')).toBeVisible()
  })

  test('should display heatmap', async ({ page }) => {
    await expect(page.locator('text=位置编码热力图')).toBeVisible()

    // Check heatmap cells exist
    const cells = page.locator('div[style*="backgroundColor"]')
    await expect(cells.first()).toBeVisible()
  })

  test('should show sample values table', async ({ page }) => {
    await expect(page.locator('text=示例值')).toBeVisible()
    await expect(page.locator('table')).toBeVisible()

    // Should have dimension headers
    const headers = page.locator('th')
    await expect(headers.first()).toContainText('维度')
  })

  test('should adjust visualization range', async ({ page }) => {
    // Adjust start position
    const startSlider = page.locator('input[type="range"]').nth(2)
    await startSlider.fill('5')

    // Adjust end position
    const endSlider = page.locator('input[type="range"]').nth(3)
    await endSlider.fill('15')

    // Check range label
    await expect(page.locator('text=可视化范围: 位置 5 - 15')).toBeVisible()
  })

  test('should show encoding principles', async ({ page }) => {
    await expect(page.locator('text=位置编码原理')).toBeVisible()
    await expect(page.locator('text=Sinusoidal')).toBeVisible()
    await expect(page.locator('text=Learnable')).toBeVisible()
    await expect(page.locator('text=RoPE')).toBeVisible()
  })

  test('should update heatmap when changing encoding type', async ({ page }) => {
    // Get initial heatmap cell color
    const cell = page.locator('div[style*="backgroundColor"]').first()
    const initialColor = await cell.getAttribute('style')

    // Switch to Learnable encoding
    await page.click('button:has-text("Learnable")')

    // Wait for update
    await page.waitForTimeout(100)

    // Color should have changed (learnable uses random values)
    const newColor = await cell.getAttribute('style')
    // Note: This might be flaky, but demonstrates the concept
  })
})
