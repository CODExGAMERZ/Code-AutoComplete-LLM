import os
import urllib.request
import zipfile
import shutil

RAW_DIR = "data/raw"

REPOS = {
    "python_algorithms": "https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip",
    "c_algorithms": "https://github.com/TheAlgorithms/C/archive/refs/heads/master.zip",
    "java_algorithms": "https://github.com/TheAlgorithms/Java/archive/refs/heads/master.zip",
    "cpython": "https://github.com/python/cpython/archive/refs/heads/3.12.zip",
    "redis": "https://github.com/redis/redis/archive/refs/heads/7.2.zip",
    "commons_lang": "https://github.com/apache/commons-lang/archive/refs/heads/master.zip"
}

# Fallback urls using 'main' or 'master'
FALLBACK_REPOS = {
    "python_algorithms": "https://github.com/TheAlgorithms/Python/archive/refs/heads/main.zip",
    "c_algorithms": "https://github.com/TheAlgorithms/C/archive/refs/heads/main.zip",
    "java_algorithms": "https://github.com/TheAlgorithms/Java/archive/refs/heads/main.zip",
    "cpython": "https://github.com/python/cpython/archive/refs/heads/main.zip",
    "redis": "https://github.com/redis/redis/archive/refs/heads/unstable.zip",
    "commons_lang": "https://github.com/apache/commons-lang/archive/refs/heads/main.zip"
}

def download_and_extract(lang, url, fallback_url):
    zip_path = f"{lang}.zip"
    print(f"Downloading {lang} dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Failed with standard branch, trying fallback {fallback_url}...")
        try:
            urllib.request.urlretrieve(fallback_url, zip_path)
        except Exception as e_fallback:
            print(f"Failed to download {lang} dataset: {e_fallback}")
            return False

    print(f"Extracting {lang} dataset...")
    dest_folder = os.path.join(RAW_DIR, lang)
    os.makedirs(dest_folder, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
        
    os.remove(zip_path)
    print(f"Completed {lang} dataset extraction.")
    return True

def main():
    if os.path.exists(RAW_DIR):
        print(f"Cleaning existing {RAW_DIR} directory...")
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)

    for lang in REPOS:
        download_and_extract(lang, REPOS[lang], FALLBACK_REPOS[lang])

    print("All datasets downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
