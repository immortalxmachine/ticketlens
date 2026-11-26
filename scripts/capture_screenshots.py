"""Capture real screenshots + a demo video of the running app with Playwright.

Needs the API on :8020 and Gradio on :7870 already running.
Run:  python scripts/capture_screenshots.py
Outputs: docs/screenshots/01-gradio.png, 01b-gradio-summarize.png, 02-api-docs.png,
         reports/videos/*.webm (converted to demo.gif by scripts/make_gif.sh)
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
VIDEOS = ROOT / "reports" / "videos"
GRADIO = "http://localhost:7870"
API = "http://localhost:8020"
VIEW = {"width": 1280, "height": 760}


def gradio_static_shots(page) -> None:
    page.goto(GRADIO)
    page.get_by_role("button", name="Classify").click()
    page.wait_for_selector("text=Category:", timeout=60000)
    page.wait_for_timeout(500)
    page.screenshot(path=SHOTS / "01-gradio.png", full_page=True)

    page.get_by_role("tab", name="Summarize conversation").click()
    page.get_by_role("button", name="Summarize").click()
    page.wait_for_selector("text=facebook/bart-large-cnn", timeout=120000)
    page.wait_for_timeout(500)
    page.screenshot(path=SHOTS / "01b-gradio-summarize.png", full_page=True)


def swagger_shot(page) -> None:
    page.goto(f"{API}/docs")
    page.get_by_text("/classify", exact=True).first.click()
    page.get_by_role("button", name="Try it out").click()
    page.get_by_role("button", name="Execute").click()
    page.wait_for_selector(".response .highlight-code", timeout=60000)
    page.locator(".live-responses-table").scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    page.screenshot(path=SHOTS / "02-api-docs.png", full_page=False)


def demo_video(browser) -> None:
    """Short scripted flow recorded as video (about 15 s)."""
    ctx = browser.new_context(viewport=VIEW, record_video_dir=str(VIDEOS), record_video_size=VIEW)
    page = ctx.new_page()
    page.goto(GRADIO)
    page.wait_for_timeout(800)
    box = page.get_by_label("Customer message")
    box.fill("")
    box.type("I need to talk to someone about a double charge on my card", delay=25)
    page.get_by_role("button", name="Classify").click()
    page.wait_for_selector("text=Category:", timeout=60000)
    page.wait_for_timeout(1800)
    page.get_by_role("tab", name="Summarize conversation").click()
    page.wait_for_timeout(600)
    page.get_by_label("flan-t5").check()
    page.get_by_role("button", name="Summarize").click()
    page.wait_for_selector("text=google/flan-t5-base", timeout=120000)
    page.wait_for_timeout(2500)
    ctx.close()  # finalises the .webm file


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEW)
        gradio_static_shots(page)
        swagger_shot(page)
        demo_video(browser)
        browser.close()
    print("done")


if __name__ == "__main__":
    main()
