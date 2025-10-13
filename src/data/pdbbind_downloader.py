import urllib.request
from tqdm import tqdm
from pathlib import Path
from utils.logger import setup_logging

logger = setup_logging()

PDBBIND_URLS = {
    "refined": "https://pdbbind-1301734146.cos.ap-shanghai.myqcloud.com/subscribe/v2020/PDBbind_v2020_refined.tar.gz?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDzoOinb9RTFkyvc3D6j5AxVmmVyAyVllV%26q-sign-time%3D1760291725%3B1760295385%26q-key-time%3D1760291725%3B1760295385%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D63674eda79e88de7db449c4540c704e85b65de84",
    "index_refined": "https://pdbbind-1301734146.cos.ap-shanghai.myqcloud.com/subscribe/v2020/PDBbind_v2020_plain_text_index.tar.gz?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDzoOinb9RTFkyvc3D6j5AxVmmVyAyVllV%26q-sign-time%3D1760291763%3B1760295423%26q-key-time%3D1760291763%3B1760295423%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D7306ebcb5870277acf58ffcb991b3210bb7a1f2b"
}

def download_file(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logger.info(f"File already exists -> {dest_path.name}, skipping.")
        return dest_path

    logger.info(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        total = int(response.info().get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest_path.name}") as pbar:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
    logger.info(f"Download complete: {dest_path}")
    return dest_path