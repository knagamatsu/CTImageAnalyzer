# CTスキャンの断面画像の解析

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 画像横幅
IMG_WIDTH = 200

st.markdown("### CTスキャンの断面画像の解析")

file = st.sidebar.file_uploader("画像ファイル")
if not file:
    st.stop()

bottom_trim = st.sidebar.slider(
    label="下部トリム（画像下部をトリミング）",
    min_value=0,
    max_value=400,
)

# ぼかして平滑化するときのサイズ（0は平滑化しない）
blur_size = st.sidebar.number_input("Blurサイズ（ぼかし用）", 0)

fat_range = st.sidebar.slider(
    label="脂肪範囲",
    min_value=0,
    max_value=128,
    value=(48, 58),
)


def set_msl_range(min_value, max_value):
    """msl_rangeに値を設定"""
    st.session_state["msl_range"] = min_value, max_value


msl_range = st.sidebar.slider(
    label="筋肉範囲",
    min_value=0,
    max_value=128,
    value=(67, 77),
    key="msl_range",
)

# 画像読み込みと加工
with tempfile.NamedTemporaryFile(delete=True) as tf:
    with open(tf.name, "wb") as fp:
        fp.write(file.read())
        file.close()
    img = cv2.imread(tf.name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレーに変換
if isinstance(bottom_trim, int) and bottom_trim > 0:  # 下部トリム
    img = img[:-bottom_trim]

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### トリム済み元画像")
    st.image(img, width=IMG_WIDTH)

if blur_size:
    img = cv2.blur(img, (blur_size, blur_size))
    with col2:
        st.markdown("#### ぼかし画像")
        st.image(img, width=IMG_WIDTH)

lo, up = fat_range[0] - 10, msl_range[1] + 10
hist = np.histogram(img.astype(int), bins=(up - lo), range=(lo, up))[0]
peak = lo + hist.argmax()
st.markdown(f"#### ヒストグラム（ピーク位置:{peak}）")
st.bar_chart(pd.Series(hist, index=range(lo, up)))

peak_margin = st.sidebar.number_input("ピーク位置前後のマージン", 5)
st.sidebar.button(
    "筋肉範囲をピーク位置から設定",
    on_click=set_msl_range,
    args=(peak - peak_margin, peak + peak_margin),
)

img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
flg1 = (fat_range[0] <= img2) & (img2 <= fat_range[1])  # 脂肪かどうか
flg2 = (msl_range[0] <= img2) & (img2 <= msl_range[1])  # 筋肉かどうか
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 脂肪範囲（黃）")
    st.image(np.where(flg1, [255, 255, 80], img2), width=IMG_WIDTH)  # 脂肪を黃に
    st.write(f"面積: {flg1.sum()}")
with col2:
    st.markdown("#### 筋肉範囲（赤）")
    st.image(np.where(flg2, [255, 80, 80], img2), width=IMG_WIDTH)  # 筋肉を赤に
    st.write(f"面積: {flg2.sum()}")

st.write(f"脂肪 / 筋肉: {flg1.sum() / flg2.sum():.4}")
