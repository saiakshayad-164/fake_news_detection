"""
DATA ACQUISITION AND CLEANING
This module downloads the authentic UCI SMS Spam Collection, reads it,
removes unusable rows and duplicate messages, and saves a clean CSV.
"""

from pathlib import Path
import io
import zipfile
import urllib.request
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_FILE = DATA_DIR / "SMSSpamCollection.tsv"
CLEAN_FILE = DATA_DIR / "sms_spam_clean.csv"

# Official UCI archive; the GitHub mirror is a fallback.
DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://github.com/veermshah/cs6350-sms-spam-dataset/raw/refs/heads/main/SMSSpamCollection",
]

def download_dataset():
    """Download the raw dataset only when it is not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 100_000:
        print("Raw dataset already exists. Skipping download.")
        return RAW_FILE

    errors = []
    for url in DATA_URLS:
        try:
            print(f"Downloading dataset from: {url}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()

            # UCI provides a ZIP archive; the mirror provides the text file.
            if zipfile.is_zipfile(io.BytesIO(content)):
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    member = next(
                        name for name in archive.namelist()
                        if name.endswith("SMSSpamCollection")
                    )
                    raw_bytes = archive.read(member)
            else:
                raw_bytes = content

            if len(raw_bytes) < 100_000:
                raise ValueError("Downloaded file is unexpectedly small.")

            RAW_FILE.write_bytes(raw_bytes)
            print(f"Dataset saved to: {RAW_FILE}")
            return RAW_FILE
        except Exception as error:
            errors.append(f"{url}: {error}")

    raise RuntimeError(
        "Could not download the UCI dataset. Check your internet connection "
        "and try again.\n" + "\n".join(errors)
    )

def load_and_clean_dataset():
    """Read labels/messages, validate rows and remove exact duplicate texts."""
    raw_path = download_dataset()

    # The original file has no header and uses a tab between label and message.
    data = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        quoting=3,
        on_bad_lines="warn",
    )

    # Standardize labels and message text.
    data["label"] = data["label"].astype("string").str.strip().str.lower()
    data["message"] = data["message"].astype("string").str.strip()

    # Keep only the two documented classes and non-empty messages.
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"].notna() & data["message"].ne("")]
    data["message"] = data["message"].astype(str)
    data = data.drop_duplicates(subset=["message"], keep="first")
    data = data.reset_index(drop=True)

    if data.empty or data["label"].nunique() != 2:
        raise ValueError("Dataset validation failed: both ham and spam are required.")

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CLEAN_FILE, index=False, encoding="utf-8")
    print(f"Clean dataset saved: {CLEAN_FILE}")
    print(f"Records after cleaning: {len(data)}")
    print("Class counts:\n", data["label"].value_counts())
    return data

if __name__ == "__main__":
    load_and_clean_dataset()
