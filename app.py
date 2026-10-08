import time
import json
import warnings
from pathlib import Path
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.wcs import FITSFixedWarning

# Skyfield 関連
from skyfield.api import load
from skyfield.data import hipparcos, stellarium

# 警告メッセージの抑制
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*Some non-standard WCS keywords were excluded.*')

# Page Configuration
st.set_page_config(page_title="Astrometry WCS Solver", page_icon="🌌", layout="wide")

# ==========================================
# 1. API Key の安全な取得 (Streamlit Secrets)
# ==========================================
def get_api_key():
    """Streamlit Secrets から API_KEY を取得"""
    if "API_KEY" in st.secrets:
        return st.secrets["API_KEY"]
    else:
        st.error("APIキーが見つかりません。Streamlit Secrets に 'API_KEY' を設定してください。")
        st.stop()

# ==========================================
# 2. Skyfield ＋ Stellarium データロード (キャッシュ化)
# ==========================================
@st.cache_data(show_spinner=False)
def load_stellarium_constellations():
    """Skyfield と Stellarium 公式データから星座線ペアとHIP星データを取得"""
    with load.open(hipparcos.URL) as f:
        stars_df = hipparcos.load_dataframe(f)

    url = 'https://raw.githubusercontent.com/Stellarium/stellarium/eb47095a9282cf6b981f6e37fe1ea3a3ae0fd167/skycultures/modern_st/constellationship.fab'
    with load.open(url) as f:
        constellations = stellarium.parse_constellations(f)

    edges = [edge for name, edges in constellations for edge in edges]
    hip_ids = sorted(list(set(sum(edges, ()))))
    valid_stars_df = stars_df.loc[stars_df.index.intersection(hip_ids)].copy()

    return valid_stars_df, edges

# ==========================================
# 3. Astrometry.net API 通信関数群
# ==========================================
def get_session_token(api_key):
    url = "https://nova.astrometry.net/api/login"
    data = {'request-json': json.dumps({"apikey": api_key})}
    response = requests.post(url, data=data)
    res_json = response.json()
    
    if res_json.get('status') != 'success':
        raise Exception(f"APIログインエラー: {res_json}")
    return res_json.get('session')

def upload_image(session_token, image_bytes):
    url = "https://nova.astrometry.net/api/upload"
    upload_args = {
        "session": session_token,
        "allow_commercial_use": "d",
        "allow_modifications": "d",
        "publicly_visible": "n"
    }
    
    files = {'file': image_bytes}
    data = {'request-json': json.dumps(upload_args)}
    response = requests.post(url, data=data, files=files)
    
    res_json = response.json()
    if res_json.get('status') != 'success':
        raise Exception(f"画像のアップロードに失敗しました: {res_json}")
    return res_json.get('subid')

def wait_for_job(subid, status_placeholder):
    url = f"https://nova.astrometry.net/api/submissions/{subid}"
    
    while True:
        try:
            res = requests.get(url).json()
        except Exception:
            time.sleep(3)
            continue
            
        jobs = res.get('jobs', [])
        if jobs and len(jobs) > 0 and jobs[0] is not None:
            job_id = jobs[0]
            job_url = f"https://nova.astrometry.net/api/jobs/{job_id}"
            
            try:
                job_res = requests.get(job_url).json()
            except Exception:
                time.sleep(3)
                continue
                
            status = job_res.get('status')
            if status == 'solving':
                status_placeholder.info("Astrometry.net で解析中（キュー処理中）...")
            elif status == 'success':
                status_placeholder.success(f"解析成功！ Job ID: {job_id}")
                return job_id
            elif status == 'failure':
                raise Exception("解析に失敗しました。星の数が不足しているかノイズが多い可能性があります。")
                
        else:
            status_placeholder.info("ジョブの割り当てを待機中...")
            
        time.sleep(4)

def get_wcs_header(job_id):
    wcs_url = f"https://nova.astrometry.net/wcs_file/{job_id}"
    res = requests.get(wcs_url)
    return res.text

# ==========================================
# 4. FITS保存 & アノテーション描画・解析
# ==========================================
def save_as_fits(image, wcs_header_str, output_fits_path):
    img_gray = image.convert('L')
    img_data = np.array(img_gray)

    wcs = WCS(header=fits.Header.fromstring(wcs_header_str), relax=True)
    clean_header = wcs.to_header()

    hdu = fits.PrimaryHDU(data=img_data, header=clean_header)
    hdu.writeto(output_fits_path, overwrite=True)
    return wcs

def draw_annotations(image, wcs, output_plot_path):
    width, height = image.size
    stars_df, edges = load_stellarium_constellations()

    center_coord = wcs.pixel_to_world(width / 2, height / 2)
    corner_coord = wcs.pixel_to_world(0, 0)
    radius = center_coord.separation(corner_coord) * 1.5

    all_coords = SkyCoord(
        ra=stars_df['ra_hours'].values * 15.0 * u.deg,
        dec=stars_df['dec_degrees'].values * u.deg,
        frame='icrs'
    )
    
    sep = center_coord.separation(all_coords)
    near_mask = sep < radius
    filtered_df = stars_df[near_mask].copy()
    filtered_coords = all_coords[near_mask]

    if len(filtered_coords) > 0:
        raw_x, raw_y = wcs.world_to_pixel(filtered_coords)
        pix_x =
