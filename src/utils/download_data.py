import os
import gdown
import zipfile
import shutil

def download_and_extract_data():
    """
    Downloads data from Google Drive and extracts it
    """
    print("Downloading data from Google Drive...")
    
    # URL for the data
    url = "https://drive.google.com/uc?id=1JT3DyD0wnz7-7i4Zf5D9Sp9Q90_iBsga"
    
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Path for the downloaded zip file
    zip_path = os.path.join(data_dir, "vivos_data.zip")
    
    # Download the file
    gdown.download(url, zip_path, quiet=False)
    
    print(f"Data downloaded to {zip_path}")
    print("Extracting data...")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"Data extracted to {data_dir}")
    
    # Clean up the zip file
    os.remove(zip_path)
    print("Zip file removed")
    
    return data_dir

if __name__ == "__main__":
    download_and_extract_data()