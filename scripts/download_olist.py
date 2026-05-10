import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATASET_ID = "olistbr/brazilian-ecommerce"
DATA_DIR = Path("data/raw")

def download_with_kaggle_cli():
    # Validate credentials exist in ~/.kaggle/kaggle.json
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        # Fallback: use env vars to create it
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if username and key:
            kaggle_path.parent.mkdir(parents=True, exist_ok=True)
            with open(kaggle_path, "w") as f:
                f.write(f'{{"username":"{username}","key":"{key}"}}')
            kaggle_path.chmod(0o600)
            print("✅ Created ~/.kaggle/kaggle.json from env vars")
        else:
            raise EnvironmentError(
                "❌ Kaggle credentials missing. Set KAGGLE_USERNAME & KAGGLE_KEY in .env "
                "or create ~/.kaggle/kaggle.json from https://www.kaggle.com/settings"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading {DATASET_ID} → {DATA_DIR}")

    # Run kaggle CLI with retry logic
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_ID,
        "-p", str(DATA_DIR),
        "--unzip"
    ]
    
    # Retry up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=120  # 2 minute timeout per attempt
            )
            print(f"✅ Download complete:\n{result.stdout}")
            break
        except subprocess.TimeoutExpired:
            print(f"⚠️ Attempt {attempt+1} timed out. Retrying...")
            if attempt == 2:
                raise
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e.stderr}")
            if attempt == 2:
                raise

    # Verify files
    csv_files = list(DATA_DIR.rglob("*.csv"))
    print(f"\n📊 Found {len(csv_files)} CSV files in {DATA_DIR}")
    for f in csv_files[:5]:  # Show first 5
        print(f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    download_with_kaggle_cli()