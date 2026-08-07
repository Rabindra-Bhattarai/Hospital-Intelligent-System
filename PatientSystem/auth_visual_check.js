const { chromium } = require('C:/Users/rabin/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright-core');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:/Users/rabin/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe'
  });
  const checks = [];
  for (const [width, height, name] of [[1366,768,'desktop-1366x768'], [1440,900,'desktop-1440x900'], [820,1180,'tablet-820x1180'], [390,844,'mobile-390x844']]) {
    const page = await browser.newPage({ viewport: { width, height } });
    await page.goto('http://127.0.0.1:8502', { waitUntil: 'networkidle' });
    await page.locator('.auth-title').waitFor({ state: 'visible', timeout: 30000 });
    const layout = await page.evaluate(() => {
      const hero = document.querySelector('.st-key-hero_panel');
      const auth = document.querySelector('.st-key-auth_region');
      const card = document.querySelector('.st-key-auth_card');
      const h = hero?.getBoundingClientRect();
      const a = auth?.getBoundingClientRect();
      const c = card?.getBoundingClientRect();
      const main = document.querySelector('[data-testid="stMain"]');
      return { hero: h && {x:h.x,y:h.y,w:h.width,h:h.height}, auth: a && {x:a.x,y:a.y,w:a.width,h:a.height}, card: c && {x:c.x,y:c.y,w:c.width,h:c.height}, mainScrollHeight: main?.scrollHeight, viewportHeight: innerHeight };
    });
    await page.screenshot({ path: `screenshots/${name}.png`, fullPage: false });
    checks.push({ name, layout });
    if (width === 1366) {
      await page.getByRole('button', { name: 'Create Account' }).click();
      await page.getByText('Step 1 of 2', { exact: false }).waitFor({ state: 'visible' });
      await page.screenshot({ path: 'screenshots/register-step1-1366x768.png', fullPage: false });
    }
    if (width === 1440) {
      await page.getByRole('button', { name: 'Create Account' }).click();
      await page.locator('.register-step').waitFor({ state: 'visible' });
      await page.screenshot({ path: 'screenshots/register-step1-1440x900.png', fullPage: false });
      await page.getByLabel('Full Name').fill('Anita Thapa');
      await page.getByLabel('Username').fill('anita.thapa');
      await page.getByRole('button', { name: 'Continue to security' }).click();
      await page.getByLabel('Password', { exact: true }).fill('Weak');
      await page.getByLabel('Password', { exact: true }).press('Enter');
      await page.getByText('Password strength', { exact: false }).waitFor({ state: 'visible' });
      await page.screenshot({ path: 'screenshots/register-step2-validation-1440x900.png', fullPage: false });
    }
    if (width === 390) {
      await page.getByRole('button', { name: 'Create Account' }).click();
      await page.locator('.register-step').waitFor({ state: 'visible' });
      await page.screenshot({ path: 'screenshots/register-step1-mobile-390x844.png', fullPage: false });
      await page.getByLabel('Full Name').fill('Anita Thapa');
      await page.getByLabel('Username').fill('anita.thapa');
      await page.waitForTimeout(750);
      await page.getByRole('button', { name: 'Continue to security' }).click();
      await page.getByText('Step 2 of 2', { exact: false }).waitFor({ state: 'visible' });
      await page.screenshot({ path: 'screenshots/register-step2-mobile-390x844.png', fullPage: false });
    }
    await page.close();
  }
  const validationPage = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  await validationPage.goto('http://127.0.0.1:8502', { waitUntil: 'networkidle' });
  await validationPage.getByRole('button', { name: 'Sign in securely' }).click();
  await validationPage.getByText('Please enter both username and password.', { exact: true }).waitFor({ state: 'visible' });
  await validationPage.close();
  console.log(JSON.stringify(checks, null, 2));
  await browser.close();
})().catch(err => { console.error(err); process.exit(1); });
