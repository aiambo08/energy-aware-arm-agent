"""Screenshots of site/index.html and a GIF of its 3D replay, through Chrome's CDP endpoint.

    uv run --with playwright python scripts/media/capture_site.py --out docs/media

Needs a running Chrome with remote debugging (``--cdp``) and ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument("--cdp", default="http://localhost:29229")
    ap.add_argument("--task", default="sort3@1")
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(a.cdp)
        ctx = browser.new_context(viewport={"width": 1200, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto((ROOT / "site" / "index.html").as_uri())
        page.wait_for_timeout(800)
        page.screenshot(path=str(a.out / "site_overview.png"))
        page.locator("#pareto").screenshot(path=str(a.out / "site_pareto.png"))
        page.select_option("#rtask", a.task)
        page.click("#play")
        view = page.locator(".view")
        view.scroll_into_view_if_needed()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(a.frames):
                v = round(1000 * i / (a.frames - 1))
                js = f"e=>{{e.value={v};e.dispatchEvent(new Event('input'))}}"
                page.eval_on_selector("#scrub", js)
                page.wait_for_timeout(60)
                view.screenshot(path=f"{tmp}/{i:05d}.png")
            pal = (
                "fps=12,scale=720:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=64[p];[b][p]paletteuse"
            )
            ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12"]
            frames = ["-i", f"{tmp}/%05d.png", "-vf", pal, str(a.out / "site_replay.gif")]
            subprocess.run([*ffmpeg, *frames], check=True)
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
