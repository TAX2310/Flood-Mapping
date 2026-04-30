import zipfile
import os

def download_and_extract(cfg):
    data_path = cfg.DATA_PATH
    zip_path = cfg.ZIP_PATH
    sentinel1_dir = cfg.S1_PATH

    # 2. Download if not already present
    if not zip_path.exists() and not sentinel1_dir.exists():
        print("⬇️ Downloading dataset...")
        os.system(f"wget -O '{zip_path}' '{cfg.ZIP_URL}'")
    else:
        print("✅ Zip or Dataset already exists, skipping download.")

    # 3. Check if already extracted

    if sentinel1_dir.exists():
        print("✅ Dataset already extracted, skipping unzip.")
    else:
        print("📦 Extracting dataset...")

        extract_path = cfg.ROOT.resolve()  # forces correct absolute path

        print(f"Extracting to: {extract_path}")

        with zipfile.ZipFile(cfg.ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print("✅ Extraction complete.")

    # 4. Delete zip to save space
    if zip_path.exists():
        zip_path.unlink()
        print("🗑️ Zip file deleted.")

    # 5. Final check
    print("\n📂 Final structure:")
    for p in data_path.iterdir():
        print(" -", p.name)

    return data_path