import { test, expect } from '@playwright/test'

/**
 * Quantization Demo E2E Tests
 */

test.describe('Quantization Demo Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/quantization')
  })

  test('should load quantization page', async ({ page }) => {
    await expect(page.locator('h2')).toContainText('INT8 量化')
    await expect(page.locator('text=测试数据配置')).toBeVisible()
  })

  test('should adjust data size slider', async ({ page }) => {
    const slider = page.locator('input[type="range"]').first()

    // Set to 500
    await slider.fill('500')

    // Check label updated
    await expect(page.locator('text=数据大小: 500')).toBeVisible()
  })

  test('should switch quantization type', async ({ page }) => {
    // Initially symmetric should be checked
    const symmetric = page.locator('input[type="radio"]').first()
    await expect(symmetric).toBeChecked()

    // Switch to asymmetric
    const asymmetric = page.locator('input[type="radio"]').last()
    await asymmetric.check()
    await expect(asymmetric).toBeChecked()
  })

  test('should generate test data', async ({ page }) => {
    await page.click('button:has-text("生成测试数据")')

    // Quantize button should be enabled
    const quantizeBtn = page.locator('button:has-text("执行量化")')
    await expect(quantizeBtn).toBeEnabled()
  })

  test('should perform quantization', async ({ page }) => {
    // Generate data
    await page.click('button:has-text("生成测试数据")')

    // Perform quantization
    await page.click('button:has-text("执行量化")')

    // Should show quantization parameters
    await expect(page.locator('text=量化参数')).toBeVisible()
    await expect(page.locator('text=Scale')).toBeVisible()
    await expect(page.locator('text=Zero Point')).toBeVisible()

    // Should show size comparison
    await expect(page.locator('text=大小对比')).toBeVisible()
    await expect(page.locator('text=压缩比')).toBeVisible()

    // Should show accuracy metrics
    await expect(page.locator('text=精度分析')).toBeVisible()
    await expect(page.locator('text=RMSE')).toBeVisible()
  })

  test('should show data comparison table', async ({ page }) => {
    // Generate and quantize
    await page.click('button:has-text("生成测试数据")')
    await page.click('button:has-text("执行量化")')

    // Should show data table
    await expect(page.locator('text=数据对比')).toBeVisible()
    await expect(page.locator('table')).toBeVisible()

    // Should have headers
    await expect(page.locator('th:has-text("原始值")')).toBeVisible()
    await expect(page.locator('th:has-text("量化值")')).toBeVisible()
    await expect(page.locator('th:has-text("误差")')).toBeVisible()
  })

  test('should quantize example model', async ({ page }) => {
    // Click quantize model button
    page.on('dialog', dialog => dialog.accept())
    await page.click('button:has-text("量化示例模型")')

    // Dialog should appear (we accepted it above)
    // Wait a bit for processing
    await page.waitForTimeout(1000)
  })

  test('should show quantization principles', async ({ page }) => {
    await expect(page.locator('text=量化原理')).toBeVisible()
    await expect(page.locator('text=对称量化')).toBeVisible()
    await expect(page.locator('text=非对称量化')).toBeVisible()
  })
})
