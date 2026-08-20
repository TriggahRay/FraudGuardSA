"""
FraudGuard SA - End-to-end test harness.

Runs the nine documented test cases against the running Flask application
and prints a results table for the testing-evidence slide.

Usage:
    1. Start the app in one terminal:   py app.py
    2. In a second terminal, run:       py run_tests.py

Missing test fixtures (.csv, oversized file, text-free PDF) are created
automatically in a tests_tmp folder and removed afterwards.
"""
import io
import os
import shutil
import sys

import requests

BASE = "http://127.0.0.1:5000"
TMP = "tests_tmp"

FRAUD_TEXT = ("URGENT HIRING!!! Work from home and earn R50,000 per month. No experience "
              "needed. Send your CV, copy of ID and banking details to secure your position. "
              "WhatsApp us NOW on 083 000 0000. Registration fee of R350 required to access "
              "training materials. LIMITED SLOTS AVAILABLE.")


def make_fixtures():
    os.makedirs(TMP, exist_ok=True)

    # .txt scam posting
    with open(f"{TMP}/test_fraud.txt", "w", encoding="utf-8") as fh:
        fh.write(FRAUD_TEXT)

    # wrong extension
    with open(f"{TMP}/test_wrong.csv", "w", encoding="utf-8") as fh:
        fh.write("title,text\na,b")

    # oversized file (6 MB)
    with open(f"{TMP}/test_big.txt", "wb") as fh:
        fh.write(b"x" * (6 * 1024 * 1024))

    # PDF with no text layer
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        c = canvas.Canvas(f"{TMP}/test_scanned.pdf", pagesize=A4)
        c.save()
    except ImportError:
        print("  (reportlab not installed - T8 will be skipped. "
              "Install with: pip install reportlab)")


def post_json(text):
    return requests.post(f"{BASE}/analyse", json={"text": text}, timeout=30)


def post_file(path):
    with open(path, "rb") as fh:
        return requests.post(f"{BASE}/analyse", files={"file": (os.path.basename(path), fh)},
                             timeout=60)


def describe(resp):
    """Turn a response into a short readable result string."""
    try:
        data = resp.json()
    except Exception:
        return f"HTTP {resp.status_code}, no JSON returned"
    if "error" in data:
        return data["error"]
    return f"{data['label']}, {data['fraud_prob']}% fraud probability"


def main():
    try:
        requests.get(BASE, timeout=10)
    except Exception:
        print("Cannot reach the app at " + BASE)
        print("Start it first with:  py app.py   (leave that window open)")
        sys.exit(1)

    make_fixtures()

    tests = [
        ("T1", "Pasted fraudulent posting", "Fraudulent",
         lambda: post_json(FRAUD_TEXT)),
        ("T2", "Uploaded .txt scam posting", "Fraudulent",
         lambda: post_file(f"{TMP}/test_fraud.txt")),
        ("T3", "Uploaded .docx developer vacancy", "Legitimate",
         lambda: post_file("test_legit.docx")),
        ("T4", "Uploaded .pdf work-from-home advert", "Suspicious",
         lambda: post_file("test_suspicious.pdf")),
        ("T5", "Uploaded .csv file", "Rejected",
         lambda: post_file(f"{TMP}/test_wrong.csv")),
        ("T6", "Nine-character input", "Rejected",
         lambda: post_json("too short")),
        ("T7", "Six megabyte file", "Rejected",
         lambda: post_file(f"{TMP}/test_big.txt")),
        ("T8", "PDF with no text layer", "Rejected",
         lambda: post_file(f"{TMP}/test_scanned.pdf")),
        ("T9", "Home page request", "Serves interface",
         lambda: requests.get(BASE, timeout=15)),
    ]

    print()
    print(f"{'#':<4}{'Input':<38}{'Expected':<18}{'Actual result':<50}{'Pass'}")
    print("-" * 118)

    passed = 0
    for tid, label, expected, run in tests:
        try:
            resp = run()
            if tid == "T9":
                actual = "Interface served" if resp.status_code == 200 else f"HTTP {resp.status_code}"
                ok = resp.status_code == 200
            else:
                actual = describe(resp)
                if expected == "Rejected":
                    ok = resp.status_code >= 400
                else:
                    ok = actual.lower().startswith(expected.lower())
        except FileNotFoundError as err:
            actual, ok = f"fixture missing: {err.filename}", False
        except Exception as err:                       # noqa: BLE001
            actual, ok = f"error: {err}", False

        passed += ok
        print(f"{tid:<4}{label:<38}{expected:<18}{actual[:48]:<50}{'PASS' if ok else 'FAIL'}")

    print("-" * 118)
    print(f"{passed} of {len(tests)} tests passed\n")

    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
