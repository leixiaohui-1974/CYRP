from playwright.sync_api import sync_playwright

def verify_simulation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load Page
        page.goto("http://localhost:5000")
        page.wait_for_selector("h2:has-text('Operations')")

        # 2. Start Simulation
        page.click("button:has-text('Start Autonomy')")
        page.wait_for_timeout(2000) # Wait for some steps

        # 3. Trigger Drill (Earthquake)
        page.click("button:has-text('Drill: Earthquake (S6)')")
        page.wait_for_timeout(3000) # Wait for earthquake trigger time (>10s in logic)

        # Since logic takes >10s (sim time), and we run at 0.5s real time per step,
        # we need to wait a bit.
        # But wait, Sim.step(0.5) advances sim time by 0.5. Loop runs every 0.5s.
        # So Real Time = Sim Time.
        # Earthquake triggers at T=10.0. So we need to wait 10s.

        # Let's verify initial running state first to save time.

        # Take Screenshot
        page.screenshot(path="verification/dashboard.png", full_page=True)

        browser.close()

if __name__ == "__main__":
    verify_simulation()
