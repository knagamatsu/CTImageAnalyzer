# CTスキャンの断面画像の解析

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

IMG_WIDTH = 200

st.markdown("### CTスキャンの断面画像の解析")

# ぼかして平滑化するときのサイズ（0は平滑化しない）
blur_size = st.sidebar.number_input("Blurサイズ", 0)

fat_range = st.sidebar.slider(
    label="脂肪範囲",
    min_value=0,
    max_value=255,
    value=(48, 58),
)

msl_range = st.sidebar.slider(
    label="筋肉範囲",
    min_value=0,
    max_value=255,
    value=(67, 77),
)

file = st.sidebar.file_uploader("画像ファイル")
if not file:
    st.stop()
with tempfile.NamedTemporaryFile(delete=True) as tf:
    with open(tf.name, "wb") as fp:
        fp.write(file.read())
        file.close()
    img = cv2.imread(tf.name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレーに変換

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 元画像")
    st.image(img, width=IMG_WIDTH)

if blur_size:
    img = cv2.blur(img, (blur_size, blur_size))
    with col2:
        st.markdown("#### ぼかし画像")
        st.image(img, width=IMG_WIDTH)

st.markdown("#### ヒストグラム")
lo, up = fat_range[0] - 10, msl_range[1] + 10
hist = np.histogram(img.astype(int), bins=(up - lo), range=(lo, up))[0]
st.bar_chart(pd.Series(hist, index=range(lo, up)))

flg1 = (fat_range[0] <= img) & (img <= fat_range[1])  # 脂肪かどうか
flg2 = (msl_range[0] <= img) & (img <= msl_range[1])  # 筋肉かどうか
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 脂肪範囲（白）")
    st.image(np.where(flg1, 255, img), width=IMG_WIDTH)  # 脂肪を白に
    st.write(f"面積: {flg1.sum()}")
with col2:
    st.markdown("#### 筋肉範囲（白）")
    st.image(np.where(flg2, 255, img), width=IMG_WIDTH)  # 筋肉を白に
    st.write(f"面積: {flg2.sum()}")

st.write(f"脂肪 / 筋肉: {flg1.sum() / flg2.sum():.4}")
