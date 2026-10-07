import subprocess
import time
from datetime import datetime

RCLONE_EXE = r"C:\rclone\rclone.exe"
LOCAL_WEIGHTS = r"E:\isles26\nnunet\data\nnunet_results"
GDRIVE_WEIGHTS = "gdrive:isles26-1f/nnunet_results"
SYNC_INTERVAL_MINUTES = 10

def sync(local, remote):
    process = subprocess.Popen(
        [
            RCLONE_EXE,
            "copy",
            local,
            remote,
            "--checksum",
            "--verbose",
            "--progress",
            "--stats", "5s",
            "--stats-one-line",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()
    return process.returncode == 0


def run():
    print("Auto-sync started. Press Ctrl+C to stop.\n")

    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*50}")
        print(f"  Sync started at {ts}")
        print(f"{'='*50}\n")

        try:
            ok = sync(LOCAL_WEIGHTS, GDRIVE_WEIGHTS)

            ts_end = datetime.now().strftime("%H:%M:%S")
            if ok:
                print(f"\n[{ts_end}] Sync complete! Next sync in {SYNC_INTERVAL_MINUTES} min...")
            else:
                print(f"\n[{ts_end}] Sync FAILED. Retrying in {SYNC_INTERVAL_MINUTES} min...")

        except Exception as e:
            print(f"\n[ERROR] {e}")

        time.sleep(SYNC_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run()