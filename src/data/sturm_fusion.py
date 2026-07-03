import zipfile
import os
import wget

def bar_progress(current, total, width=80):
    progress = current / total * 100
    print(f"\rDownloading: {progress:.1f}% [{current}/{total} bytes]", end="")

def download_and_extract_dataset(cfg):
    url = cfg.DATASET_ZIP_URL
    data_path = cfg.DATA_PATH
    dataset_zip_path = cfg.DATASET_ZIP_PATH
    sentinel1_dir = cfg.S1_PATH
    mask_dir = cfg.MASK_PATH

    os.makedirs(data_path, exist_ok=True)
    is_extracted = sentinel1_dir.exists() and mask_dir.exists()

    # 2. Download if not already present
    if not dataset_zip_path.exists() and not is_extracted:
        print("Downloading dataset...")
        wget.download(url, bar=bar_progress)
    else:
        print("Zip or Dataset already exists, skipping download.")

    # 3. Check if already extracted

    if is_extracted:
        print("Dataset already extracted, skipping unzip.")
    else:
        print("Extracting dataset...")

        extract_path = cfg.ROOT.resolve()  # forces correct absolute path

        print(f"Extracting to: {extract_path}")

        with zipfile.ZipFile(cfg.DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print("Extraction complete.")

    # 4. Delete zip to save space
    if dataset_zip_path.exists():
        dataset_zip_path.unlink()
        print("Zip file deleted.")

    # 5. Final check
    print("\nFinal structure:")
    for p in data_path.iterdir():
        print(" -", p.name)

    return data_path

def download_and_extract_results(cfg):
    url = cfg.RESULTS_ZIP_URL
    results_zip_path = cfg.RESULTS_ZIP_PATH
    experiments_dir = cfg.EXP_DIR
    test_results_dir = cfg.TEST_RESULTS_DIR

    is_extracted = test_results_dir.exists() and experiments_dir.exists()

    # 2. Download if not already present
    if not results_zip_path.exists() and not is_extracted:
        print("Downloading results...")
        wget.download(url, bar=bar_progress)
    else:
        print("Zip or Results already exists, skipping download.")

    # 3. Check if already extracted

    if is_extracted:
        print("Results already extracted, skipping unzip.")
    else:
        print("Extracting results...")

        extract_path = cfg.ROOT.resolve()  # forces correct absolute path

        print(f"Extracting to: {extract_path}")

        with zipfile.ZipFile(cfg.RESULTS_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print("Extraction complete.")

    # 4. Delete zip to save space
    if results_zip_path.exists():
        results_zip_path.unlink()
        print("Zip file deleted.")

    # 5. Final check
    print("\nFinal structure:")
    for p in experiments_dir.iterdir():
        print(" -", p.name)
    for p in test_results_dir.iterdir():
        print(" -", p.name)