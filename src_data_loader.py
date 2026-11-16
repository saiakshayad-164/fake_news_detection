from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FAKE_PATH = DATA_DIR / "Fake.csv"
REAL_PATH = DATA_DIR / "True.csv"
DATA_URLS = {
    FAKE_PATH: "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/Fake.csv",
    REAL_PATH: "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/True.csv",
}

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in DATA_URLS.items():
        if path.exists() and path.stat().st_size > 1000:
            print(f"{path.name} already exists; skipping download.")
            continue
        print(f"Downloading {path.name}...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(path, "wb") as output:
            output.write(response.read())
        if path.stat().st_size < 1000:
            raise RuntimeError(f"Downloaded {path.name} appears invalid.")
        print(f"Saved {path}")

def load_and_clean_dataset():
    download_dataset()
    fake = pd.read_csv(FAKE_PATH)
    real = pd.read_csv(REAL_PATH)
    for frame in (fake, real):
        if "text" not in frame.columns:
            raise ValueError("Each source CSV must contain a text column.")
        if "title" not in frame.columns:
            frame["title"] = ""
    fake["label"] = "Fake"
    real["label"] = "Real"
    data = pd.concat([fake, real], ignore_index=True)
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["content"] = (data["title"].str.strip() + " " + data["text"].str.strip()).str.strip()
    data["content"] = data["content"].str.replace(r"\s+", " ", regex=True)
    data = data[data["content"].str.len() >= 30].drop_duplicates(subset="content").reset_index(drop=True)
    data = data[["title", "text", "content", "label"]]
    clean_path = DATA_DIR / "news_clean.csv"
    data.to_csv(clean_path, index=False)
    print(f"Clean dataset saved: {clean_path}")
    print(f"Records after cleaning: {len(data)}")
    print("Class counts:\n", data["label"].value_counts())
    return data

if __name__ == "__main__":
    load_and_clean_dataset()
