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
from astropy.wcs.wcs import FITSFixedWarning

# 警告メッセージの抑制
warnings.simplefilter('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', message='.*Some non-standard WCS keywords were excluded.*')

# ページ設定
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
# 2. Astrometry.net API 通信関数群
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
# 3. FITS保存 & グリッド画像の生成 (高速・安全)
# ==========================================
def save_as_fits(image, wcs_header_str, output_fits_path):
    img_gray = image.convert('L')
    img_data = np.array(img_gray)

    wcs = WCS(header=fits.Header.fromstring(wcs_header_str), relax=True)
    clean_header = wcs.to_header()

    hdu = fits.PrimaryHDU(data=img_data, header=clean_header)
    hdu.writeto(output_fits_path, overwrite=True)
    return wcs

def format_wcs_header_to_lines(header_str):
    """FITSヘッダーの文字列を80文字ごとに分割して改行を入れる"""
    lines = [header_str[i:i+80] for i in range(0, len(header_str), 80)]
    return "\n".join(lines)

def draw_wcs_grid(image, wcs, output_plot_path):
    width, height = image.size

    fig = plt.figure(figsize=(10, 10 * (height / width)))
    ax = fig.add_subplot(111, projection=wcs)
    
    ax.imshow(image, origin='lower')

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # 座標グリッドの描画
    ax.coords.grid(True, color='cyan', alpha=0.5, linestyle='solid')
    ax.coords[0].set_axislabel('RA (J2000)')
    ax.coords[1].set_axislabel('Dec (J2000)')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

# ==========================================
# 4. Streamlit メイン UI
# ==========================================
st.title("Astrometry.net Cloud Plate Solver")
st.write("Astrometry.net APIを利用してPlate Solvingを行い、グリッド付きのWCS画像を高速に生成します。")

api_key = get_api_key()

uploaded_file = st.file_uploader("解析する星空画像を選択してください", type=["jpg", "jpeg", "png", "tif", "fits"])

file_id = f"{uploaded_file.name}-{uploaded_file.size}" if uploaded_file else None
if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
    st.session_state.current_file_id = file_id
    st.session_state.is_solved = False

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="アップロード画像", use_container_width=True)

    if st.button("Run Plate Solving", type="primary"):
        status_placeholder = st.empty()
        
        try:
            status_placeholder.info("Astrometry.net にログイン中...")
            session_token = get_session_token(api_key)
            
            status_placeholder.info("画像をアップロード中...")
            uploaded_file.seek(0)
            subid = upload_image(session_token, uploaded_file.getvalue())
            
            job_id = wait_for_job(subid, status_placeholder)
            
            status_placeholder.info("WCSヘッダーを取得中...")
            wcs_header_str = get_wcs_header(job_id)
            
            output_fits_path = "output_wcs.fits"
            output_plot_path = "wcs_grid_sky.png"
            output_hdr_path = "wcs_header.txt"

            formatted_header = format_wcs_header_to_lines(wcs_header_str)
            with open(output_hdr_path, "w", encoding="utf-8") as f:
                f.write(formatted_header)
            
            wcs = save_as_fits(image, wcs_header_str, output_fits_path)

            status_placeholder.info("グリッド画像を生成中...")
            draw_wcs_grid(image, wcs, output_plot_path)

            width, height = image.size
            center_coord = wcs.pixel_to_world(width / 2, height / 2)
            ra_str = center_coord.ra.to_string(unit=u.hour, sep='hms', precision=1)
            dec_str = center_coord.dec.to_string(unit=u.deg, sep='dms', precision=1)

            coord_tl = wcs.pixel_to_world(0, height)
            coord_tr = wcs.pixel_to_world(width, height)
            coord_bl = wcs.pixel_to_world(0, 0)
            coord_br = wcs.pixel_to_world(width, 0)

            fov_w_deg = (coord_bl.separation(coord_br).deg + coord_tl.separation(coord_tr).deg) / 2.0
            fov_h_deg = (coord_bl.separation(coord_tl).deg + coord_br.separation(coord_br).deg) / 2.0
            fov_diag_deg = coord_bl.separation(coord_tr).deg
            scale_arcsec = (fov_w_deg * 3600.0) / width

            st.session_state.is_solved = True
            st.session_state.ra_str = ra_str
            st.session_state.dec_str = dec_str
            st.session_state.fov_w_deg = fov_w_deg
            st.session_state.fov_h_deg = fov_h_deg
            st.session_state.fov_diag_deg = fov_diag_deg
            st.session_state.scale_arcsec = scale_arcsec
            st.session_state.output_plot_path = output_plot_path
            st.session_state.output_fits_path = output_fits_path
            st.session_state.output_hdr_path = output_hdr_path

            status_placeholder.success("全処理が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    if st.session_state.get("is_solved", False):
        with col2:
            st.image(st.session_state.output_plot_path, caption="解析結果 (WCSグリッド表示)", use_container_width=True)
        
        st.markdown("---")
        st.subheader("解析データ")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("**撮影中心座標**")
            st.write(f"- 赤経 (RA): `{st.session_state.ra_str}`")
            st.write(f"- 赤緯 (Dec): `{st.session_state.dec_str}`")
        
        with res_col2:
            st.markdown("**視野角（FOV）& 画像領域情報**")
            st.write(f"- 視野サイズ: `{st.session_state.fov_w_deg:.2f}° × {st.session_state.fov_h_deg:.2f}°` (対角: `{st.session_state.fov_diag_deg:.2f}°`)")
            st.write(f"- ピクセルスケール: `{st.session_state.scale_arcsec:.2f} arcsec/pixel`")

        st.markdown("---")
        st.subheader("ダウンロード")

        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            with open(st.session_state.output_plot_path, "rb") as f:
                st.download_button(
                    label="グリッド付き画像をダウンロード (PNG)",
                    data=f,
                    file_name=f"{Path(uploaded_file.name).stem}_grid.png",
                    mime="image/png"
                )

        with d_col2:
            with open(st.session_state.output_fits_path, "rb") as f:
                st.download_button(
                    label="WCS入り FITS をダウンロード",
                    data=f,
                    file_name=f"{Path(uploaded_file.name).stem}_wcs.fits",
                    mime="application/fits"
                )

        with d_col3:
            with open(st.session_state.output_hdr_path, "rb") as f:
                st.download_button(
                    label="WCS ヘッダー情報をダウンロード (TXT)",
                    data=f,
                    file_name=f"{Path(uploaded_file.name).stem}_header.txt",
                    mime="text/plain"
                )
