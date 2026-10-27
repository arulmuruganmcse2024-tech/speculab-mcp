"""Automated High-Quality Devpost Gallery Capture for SpecuLab-MCP.

Executes Stage 7:
1. Launches Streamlit app on port 8599 in background.
2. Uses Playwright Chromium sized to 1920x1080 (16:9 widescreen).
3. Automates capture of 10 showcase images into assets/gallery/.
4. Gracefully terminates the background Streamlit process.
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# 1. Assert Playwright installation
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("\n[ERROR] Playwright is not installed in the active environment.")
    print("Please run:")
    print("    pip install playwright")
    print("    playwright install chromium\n")
    sys.exit(1)


PORT = 8599
APP_URL = f"http://localhost:{PORT}"
GALLERY_DIR = Path("assets/gallery")


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll URL until server responds with 200 OK or timeout expires."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def terminate_process(proc: subprocess.Popen):
    """Cleanly terminate background process on Windows or Linux."""
    if proc.poll() is None:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def wait_for_streamlit_idle(page, timeout: int = 45000):
    """Wait until Streamlit status widget is idle / hidden."""
    try:
        page.locator('text=Running Autonomous Active Vision Metrology Cycle').wait_for(state="hidden", timeout=timeout)
    except Exception:
        pass

    try:
        page.wait_for_function(
            '() => !document.querySelector("[data-testid=\\"stStatusWidget\\"]") || '
            'document.querySelector("[data-testid=\\"stStatusWidget\\"]").innerText.indexOf("RUNNING") === -1',
            timeout=timeout,
        )
    except Exception:
        pass


def select_specimen(page, specimen_name: str):
    """Select a specimen in Streamlit's sidebar selectbox."""
    print(f"[*] Selecting '{specimen_name}' in sidebar...")
    sb = page.locator('div[data-testid="stSidebar"] div[data-testid="stSelectbox"]')
    if sb.count() == 0:
        sb = page.locator('div[data-testid="stSelectbox"]:has-text("Select Specimen")')

    sb.first.click()
    page.wait_for_timeout(600)

    opt = page.locator(f'li[role="option"]:has-text("{specimen_name}")')
    if opt.count() > 0:
        opt.first.click()
    else:
        page.keyboard.type(specimen_name)
        page.wait_for_timeout(300)
        page.keyboard.press("Enter")

    page.wait_for_timeout(1800)
    wait_for_streamlit_idle(page)


def execute_cycle(page, specimen_name: str):
    """Click Execute button and wait until the cycle for this specimen finishes."""
    print(f"[*] Triggering Active Multi-View Metrology Cycle for {specimen_name}...")
    btn = page.locator('button:has-text("Execute Active Multi-View Metrology Cycle"), button:has-text("Execute")')
    btn.first.click()

    try:
        page.locator('text=Running Autonomous Active Vision Metrology Cycle').wait_for(state="visible", timeout=4000)
    except Exception:
        pass

    page.locator('text=Running Autonomous Active Vision Metrology Cycle').wait_for(state="hidden", timeout=60000)
    page.locator('text=Active Inspection Complete!').wait_for(state="visible", timeout=60000)

    wait_for_streamlit_idle(page)
    page.wait_for_timeout(2500)


def main():
    print("=" * 70)
    print("  SpecuLab-MCP Stage 7: Automated High-Res Devpost Gallery Capture")
    print("=" * 70)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    cmd = [
        python_bin,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]

    print(f"[*] Launching Streamlit subprocess on port {PORT}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        print("[*] Waiting for Streamlit server to become responsive...")
        if not wait_for_server(APP_URL, timeout=30.0):
            print(f"[ERROR] Streamlit failed to respond at {APP_URL} within 30 seconds.")
            terminate_process(proc)
            sys.exit(1)
        print(f"[+] Streamlit successfully responsive at {APP_URL}!")

        with sync_playwright() as p:
            print("[*] Launching Chromium browser at 1920x1080...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
            )
            page = context.new_page()

            page.goto(APP_URL, wait_until="networkidle")
            page.wait_for_timeout(2000)

            # -------------------------------------------------------------
            # Capture 01: Hero Wow Moment (specimen_06 - Overlap Trap)
            # -------------------------------------------------------------
            print("\n[1/10] Capturing 01_hero_wow_moment.png...")
            select_specimen(page, "specimen_06")
            execute_cycle(page, "specimen_06")

            # Frame the 4 KPI metrics and the 3 comparison columns
            page.evaluate("window.scrollTo(0, 220)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "01_hero_wow_moment.png"))
            print("  -> Saved assets/gallery/01_hero_wow_moment.png")

            # -------------------------------------------------------------
            # Capture 02: Agentic Timeline Trace (specimen_06)
            # -------------------------------------------------------------
            print("\n[2/10] Capturing 02_agentic_timeline_trace.png...")
            timeline_header = page.locator('text=Dynamic Agent Decision Timeline')
            timeline_header.scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "02_agentic_timeline_trace.png"))
            print("  -> Saved assets/gallery/02_agentic_timeline_trace.png")

            # -------------------------------------------------------------
            # Capture 03: Supervisor CloudWatch Gate (Quarantine Approval)
            # -------------------------------------------------------------
            print("\n[3/10] Capturing 03_supervisor_cloudwatch_gate.png...")
            quarantine_btn = page.locator('button:has-text("Confirm Quarantine"), button:has-text("Approve Quarantine")')
            if quarantine_btn.count() > 0:
                quarantine_btn.first.scroll_into_view_if_needed()
                page.wait_for_timeout(400)
                quarantine_btn.first.click()
                page.locator('text=Quarantine event logged to CloudWatch').wait_for(state="visible", timeout=10000)
                page.wait_for_timeout(1500)
            else:
                print("  [WARN] Quarantine button not visible on current specimen")

            # Frame the confirmation banner, supervisor inputs and CloudWatch ledger
            ledger = page.locator('text=CloudWatch Audit Events Ledger')
            if ledger.count() > 0:
                ledger.first.scroll_into_view_if_needed()
                page.evaluate("window.scrollBy(0, -30)")
                page.wait_for_timeout(800)
            page.screenshot(path=str(GALLERY_DIR / "03_supervisor_cloudwatch_gate.png"))
            print("  -> Saved assets/gallery/03_supervisor_cloudwatch_gate.png")

            # -------------------------------------------------------------
            # Capture 04: Two-Pass Fusion (specimen_08 - 2-Pass Fusion)
            # -------------------------------------------------------------
            print("\n[4/10] Capturing 04_two_pass_fusion.png...")
            select_specimen(page, "specimen_08")

            # Adjust glare gate slider to 0.5% so multi-pass fusion is required for specimen_08
            slider = page.locator('div[data-testid="stSidebar"] div[role="slider"]')
            if slider.count() > 0:
                slider.first.focus()
                for _ in range(3):
                    page.keyboard.press("ArrowLeft")
                page.wait_for_timeout(800)
                wait_for_streamlit_idle(page)

            execute_cycle(page, "specimen_08")

            # Scroll so the 3 columns showing "Views Fused: 2/2" are centered
            col3 = page.locator('.active-healed-container')
            col3.first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, -120)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "04_two_pass_fusion.png"))
            print("  -> Saved assets/gallery/04_two_pass_fusion.png")

            # Reset slider back to default 2.0%
            if slider.count() > 0:
                slider.first.focus()
                for _ in range(3):
                    page.keyboard.press("ArrowRight")
                page.wait_for_timeout(800)
                wait_for_streamlit_idle(page)

            # -------------------------------------------------------------
            # Capture 05: Unresolved Escalation (specimen_10 - Omnidirectional Glare)
            # -------------------------------------------------------------
            print("\n[5/10] Capturing 05_unresolved_escalation.png...")
            select_specimen(page, "specimen_10")
            execute_cycle(page, "specimen_10")

            # Scroll to center the amber ESCALATE_UNRESOLVED_OCCLUSION warning banner
            warn_elem = page.locator('text=ESCALATE_UNRESOLVED_OCCLUSION')
            if warn_elem.count() > 0:
                warn_elem.first.scroll_into_view_if_needed()
                page.evaluate("window.scrollBy(0, -80)")
                page.wait_for_timeout(1000)
            else:
                page.evaluate("window.scrollTo(0, 350)")
                page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "05_unresolved_escalation.png"))
            print("  -> Saved assets/gallery/05_unresolved_escalation.png")

            # -------------------------------------------------------------
            # Capture 06: Clean Foil Handling Wrinkle Rejection (specimen_14)
            # -------------------------------------------------------------
            print("\n[6/10] Capturing 06_clean_foil_rejection.png...")
            select_specimen(page, "specimen_14")
            execute_cycle(page, "specimen_14")

            col3 = page.locator('.active-healed-container')
            col3.first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, -120)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "06_clean_foil_rejection.png"))
            print("  -> Saved assets/gallery/06_clean_foil_rejection.png")

            # -------------------------------------------------------------
            # Switch to Tab 2: 60-Second Judge Quick-Verify & Benchmarks
            # -------------------------------------------------------------
            print("\n[*] Switching to Tab 2 ('60-Second Judge Quick-Verify & Benchmarks')...")
            tab2 = page.locator('[role="tab"]:has-text("Quick-Verify"), [data-testid="stTab"]:has-text("Quick-Verify")')
            tab2.first.click()
            page.wait_for_timeout(1500)

            # -------------------------------------------------------------
            # Capture 07: Evaluation Metrics (6-Method Table)
            # -------------------------------------------------------------
            print("\n[7/10] Capturing 07_evaluation_metrics.png...")
            eval_title = page.locator('text=6-Method Held-Out Evaluation Benchmark')
            eval_title.first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, -30)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "07_evaluation_metrics.png"))
            print("  -> Saved assets/gallery/07_evaluation_metrics.png")

            # -------------------------------------------------------------
            # Capture 08: Hardware SIMD Benchmark (Plotly Chart & Tables)
            # -------------------------------------------------------------
            print("\n[8/10] Capturing 08_hardware_simd_benchmark.png...")
            chart_elem = page.locator('div[data-testid="stPlotlyChart"], .js-plotly-plot')
            if chart_elem.count() > 0:
                chart_elem.first.scroll_into_view_if_needed()
                page.evaluate("window.scrollBy(0, -100)")
            else:
                bench_title = page.locator('text=Authentic Hardware Execution Benchmarks')
                bench_title.first.scroll_into_view_if_needed()
                page.evaluate("window.scrollBy(0, 100)")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(GALLERY_DIR / "08_hardware_simd_benchmark.png"))
            print("  -> Saved assets/gallery/08_hardware_simd_benchmark.png")

            # -------------------------------------------------------------
            # Capture 09: COOL Verification Logs (aarch64 / OpenCV 5.0.0)
            # -------------------------------------------------------------
            print("\n[9/10] Capturing 09_cool_verification_logs.png...")
            cool_title = page.locator('text=AWS COOL Graviton Verification Log')
            cool_title.first.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, -40)")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(GALLERY_DIR / "09_cool_verification_logs.png"))
            print("  -> Saved assets/gallery/09_cool_verification_logs.png")

            # Close browser
            browser.close()

        # -------------------------------------------------------------
        # Capture 10: Printable Optical Card (Direct File Copy)
        # -------------------------------------------------------------
        print("\n[10/10] Copying 10_printable_optical_card.png...")
        src_card = Path("assets/printable_test_card.png")
        dst_card = GALLERY_DIR / "10_printable_optical_card.png"
        if src_card.exists():
            shutil.copyfile(src_card, dst_card)
            print(f"  -> Successfully copied {src_card} to {dst_card}")
        else:
            print(f"  [ERROR] Source {src_card} does not exist!")

    finally:
        print("\n[*] Tearing down Streamlit background subprocess...")
        terminate_process(proc)
        print("[+] Streamlit process terminated cleanly.")

    # Verification of all 10 files
    print("\n" + "=" * 70)
    print("  Gallery Verification Summary:")
    print("=" * 70)
    expected_files = [
        "01_hero_wow_moment.png",
        "02_agentic_timeline_trace.png",
        "03_supervisor_cloudwatch_gate.png",
        "04_two_pass_fusion.png",
        "05_unresolved_escalation.png",
        "06_clean_foil_rejection.png",
        "07_evaluation_metrics.png",
        "08_hardware_simd_benchmark.png",
        "09_cool_verification_logs.png",
        "10_printable_optical_card.png",
    ]

    all_present = True
    for f in expected_files:
        fpath = GALLERY_DIR / f
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"  [OK] {f} ({fpath.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [MISSING] {f}")
            all_present = False

    if all_present:
        print("\n[SUCCESS] All 10 high-resolution Devpost gallery images captured perfectly!")
    else:
        print("\n[WARNING] Some gallery captures are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
