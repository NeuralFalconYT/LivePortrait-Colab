import os
import urllib
import shutil
import subprocess
from tqdm import tqdm

def conditional_download(url, download_file_path):
    print(f"Downloading {os.path.basename(download_file_path)}")
    base_path = os.path.dirname(download_file_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if os.path.exists(download_file_path):
        os.remove(download_file_path)

    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError as e:
        print(f"Error: Unable to open the URL - {url}")
        print(f"Reason: {e.reason}")
        return

    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))
        except urllib.error.URLError as e:
            print(f"Error: Failed to download the file from the URL - {url}")
            print(f"Reason: {e.reason}")
            return

    print(f"Download successful!")
    print(f"URL: {url}")
    print(f"Save at: {download_file_path}")

def download_models(base_path):
    conditional_download("https://raw.githubusercontent.com/neuralfalconbackup/LivePortrait-Colab/main/webapp.py", f"{base_path}/LivePortrait/webapp.py")
    
    def download_files(file_list, relative_path, base_url):
        for file_name in file_list:
            download_file_path = f"{base_path}/{relative_path}/{file_name}"
            conditional_download(f"{base_url}/{file_name}", download_file_path)

    buffalo_l_files = ["2d106det.onnx", "det_10g.onnx"]
    buffalo_l_path = "LivePortrait/pretrained_weights/insightface/models/buffalo_l"
    buffalo_l_url = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/insightface/models/buffalo_l"

    base_models_files = [
        'appearance_feature_extractor.pth',
        'motion_extractor.pth',
        'spade_generator.pth',
        'warping_module.pth'
    ]
    base_models_path = "LivePortrait/pretrained_weights/liveportrait/base_models"
    base_models_url = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/base_models"

    retargeting_models_files = ['stitching_retargeting_module.pth']
    retargeting_models_path = "LivePortrait/pretrained_weights/liveportrait/retargeting_models"
    retargeting_models_url = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait/retargeting_models"

    landmark_file = ["landmark.onnx"]
    landmark_path = "LivePortrait/pretrained_weights/liveportrait"
    landmark_url = "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait"

    download_files(buffalo_l_files, buffalo_l_path, buffalo_l_url)
    download_files(base_models_files, base_models_path, base_models_url)
    download_files(retargeting_models_files, retargeting_models_path, retargeting_models_url)
    download_files(landmark_file, landmark_path, landmark_url)

def main():
    repo_url = "https://github.com/KwaiVGI/LivePortrait.git"
    repo_dir = "LivePortrait"
    
    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url])

    root_path = os.getcwd()
    base_path = f"{root_path}"
    download_models(base_path)
    
    os.chdir(f"{base_path}/{repo_dir}")
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    from IPython.display import clear_output
    clear_output()

if __name__ == "__main__":
    main()
