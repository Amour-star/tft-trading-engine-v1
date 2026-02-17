"""Start the dashboard (API + Streamlit)."""
import sys
sys.path.insert(0, ".")

import subprocess
import threading
from dashboard.api import start_api
from config.settings import settings


def run_streamlit():
    subprocess.run([
        "streamlit", "run", "dashboard/streamlit_app.py",
        "--server.port", str(settings.dashboard.dashboard_port),
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    # Start API in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    # Start Streamlit (foreground)
    run_streamlit()
