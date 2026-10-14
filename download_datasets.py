import os
import sys
import urllib.request
import zipfile
import tarfile
import numpy as np

DATASETS_DIR = os.path.abspath("datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def download_with_progress(url, dest_path, desc=""):
    print(f"\n[DOWNLOAD] {desc}: {url}")
    print(f"Target: {dest_path}")
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[EXISTS] File already exists ({size_mb:.1f} MB). Skipping download.")
        return

    tmp_path = dest_path + ".tmp"
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                downloaded_mb = (count * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading {desc}... {percent:3d}% ({downloaded_mb:6.1f} MB / {total_mb:6.1f} MB)")
            else:
                downloaded_mb = (count * block_size) / (1024 * 1024)
                sys.stdout.write(f"\rDownloading {desc}... {downloaded_mb:6.1f} MB")
            sys.stdout.flush()

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(tmp_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', -1))
            block_size = 1024 * 1024 # 1 MB chunks
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    sys.stdout.write(f"\rDownloading {desc}... {percent:3d}% ({downloaded / 1024 / 1024:6.1f} MB / {total_size / 1024 / 1024:6.1f} MB)")
                else:
                    sys.stdout.write(f"\rDownloading {desc}... {downloaded / 1024 / 1024:6.1f} MB")
                sys.stdout.flush()
        
        print("\nDownload complete.")
        os.rename(tmp_path, dest_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download {url}: {e}")

def parse_fvecs(filename):
    """Read .fvecs file format into numpy float32 array."""
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view('float32')

def setup_sift():
    sift_npy = os.path.join(DATASETS_DIR, "sift.npy")
    if os.path.exists(sift_npy):
        print(f"[EXISTS] sift.npy already present ({os.path.getsize(sift_npy) / (1024*1024):.1f} MB)")
        return

    tar_path = os.path.join(DATASETS_DIR, "siftsmall.tar.gz")
    url = "http://corpus-texmex.irisa.fr/texmex/siftsmall.tar.gz"
    download_with_progress(url, tar_path, desc="SIFT Dataset Archive")

    print("[EXTRACT] Extracting siftsmall.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATASETS_DIR)
    
    base_fvecs = os.path.join(DATASETS_DIR, "siftsmall", "siftsmall_base.fvecs")
    if os.path.exists(base_fvecs):
        print("[PARSE] Converting siftsmall_base.fvecs to sift.npy...")
        vectors = parse_fvecs(base_fvecs)
        np.save(sift_npy, vectors.astype(np.float32))
        print(f"[SAVED] {sift_npy} created with shape {vectors.shape} (float32).")

def setup_year_prediction_msd():
    msd_txt = os.path.join(DATASETS_DIR, "YearPredictionMSD.txt")
    if os.path.exists(msd_txt):
        print(f"[EXISTS] YearPredictionMSD.txt already present ({os.path.getsize(msd_txt) / (1024*1024):.1f} MB)")
        return

    zip_path = os.path.join(DATASETS_DIR, "YearPredictionMSD.txt.zip")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    download_with_progress(url, zip_path, desc="YearPredictionMSD Archive (~200 MB)")

    print("[EXTRACT] Extracting YearPredictionMSD.txt.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASETS_DIR)
    print(f"[EXTRACTED] YearPredictionMSD.txt ready in {DATASETS_DIR}")
    if os.path.exists(zip_path):
        os.remove(zip_path)

def setup_raster():
    raster_npy = os.path.join(DATASETS_DIR, "raster.npy")
    if os.path.exists(raster_npy):
        print(f"[EXISTS] raster.npy already present ({os.path.getsize(raster_npy) / (1024*1024):.1f} MB)")
        return
    print("[GENERATE] Generating 2048 x 2048 high-resolution raster benchmark array...")
    np.random.seed(42)
    # Generate structured spatial patterns (smooth gradient + noise + Gaussian features)
    x = np.linspace(-3, 3, 2048, dtype=np.float32)
    y = np.linspace(-3, 3, 2048, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    raster = np.sin(X**2 + Y**2) + 0.1 * np.random.randn(2048, 2048).astype(np.float32)
    np.save(raster_npy, raster)
    print(f"[SAVED] {raster_npy} created (2048x2048 float32, ~16 MB).")

def setup_higgs():
    higgs_gz = os.path.join(DATASETS_DIR, "HIGGS.csv.gz")
    higgs_csv = os.path.join(DATASETS_DIR, "HIGGS.csv")
    if os.path.exists(higgs_gz) or os.path.exists(higgs_csv):
        print(f"[EXISTS] HIGGS dataset already present.")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    print("\n" + "="*70)
    print("NOTE: HIGGS is a large dataset (~2.8 GB compressed, ~8.8 GB uncompressed).")
    print("Beginning download to datasets/HIGGS.csv.gz...")
    print("="*70)
    download_with_progress(url, higgs_gz, desc="HIGGS.csv.gz (~2.8 GB)")
    print(f"[READY] HIGGS dataset stored at {higgs_gz}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("="*70)
    print(f"  DATASET DOWNLOAD & SETUP (Target: {target})")
    print("="*70)

    if target in ("all", "sift"):
        setup_sift()

    if target in ("all", "raster"):
        setup_raster()

    if target in ("all", "msd"):
        setup_year_prediction_msd()

    if target in ("all", "higgs"):
        setup_higgs()

    print("\n" + "="*70)
    print("DATASET SETUP COMPLETED SUCCESSFULLY!")
    print("="*70)
