"""
Dashboard Launcher Script for Vector Database Comparative Analysis.
Starts the Streamlit dashboard on http://localhost:8501
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(PROJECT_ROOT, "dashboard", "app.py")

def main():
    print("=" * 70)
    print("LAUNCHING VECTOR DATABASE ANALYTICS DASHBOARD")
    print(f"Target App: {APP_PATH}")
    print("URL: http://localhost:8501")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_PATH,
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\nDashboard shutdown requested by user.")

if __name__ == "__main__":
    main()
