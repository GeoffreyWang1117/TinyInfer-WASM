import { test, expect } from '@playwright/test'

/**
 * Home Page E2E Tests
 */

test.describe('Home Page', () => {
  test('should load home page successfully', async ({ page }) => {
    await page.goto('/')

    // Check title
    await expect(page).toHaveTitle(/TinyInfer-WASM/)

    // Check main heading
    await expect(page.locator('h1')).toContainText('TinyInfer-WASM')

    // Check WASM status indicator
    await expect(page.locator('text=WASM:')).toBeVisible()
  })

  test('should show WASM loaded status', async ({ page }) => {
    await page.goto('/')

    // Wait for WASM to load (max 5 seconds)
    await expect(page.locator('text=已加载')).toBeVisible({ timeout: 5000 })
  })

  test('should navigate to different pages', async ({ page }) => {
    await page.goto('/')

    // Test navigation links
    const links = [
      { text: '图像分类', url: '/image-classification' },
      { text: '文本嵌入', url: '/text-embedding' },
      { text: '性能测试', url: '/benchmark' },
      { text: 'Worker 演示', url: '/worker' },
      { text: '量化', url: '/quantization' },
      { text: '位置编码', url: '/positional' },
    ]

    for (const link of links) {
      await page.click(`text=${link.text}`)
      await expect(page).toHaveURL(new RegExp(link.url))
      await page.goBack()
    }
  })

  test('should have responsive layout', async ({ page }) => {
    // Test desktop
    await page.setViewportSize({ width: 1920, height: 1080 })
    await page.goto('/')
    await expect(page.locator('nav')).toBeVisible()

    // Test mobile
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page.locator('nav')).toBeVisible()
  })
})
