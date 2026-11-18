"""Giao diện Streamlit cho bộ phân loại chó mèo PyTorch."""
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from common import load_classifier, predict_top_k


st.set_page_config(page_title="Phân loại giống chó mèo", page_icon="🐾", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#f2f3f5;}
.block-container {max-width:1500px;padding-top:1rem;padding-bottom:2rem;}
.topbar {min-height:82px;display:grid;grid-template-columns:1fr 3fr 1fr;align-items:center;padding:12px 24px;margin-bottom:10px;border:1px solid #e6e8ec;border-radius:18px;background:#fff;box-shadow:0 3px 12px #18223012;}
.brand {color:#344b7d;font-size:1.05rem;font-weight:700;}.brand small {display:block;color:#68738b;font-size:.78rem;font-weight:400;}
.page-title {color:#526fce;text-align:center;font-size:1.65rem;font-weight:700;}.paw {color:#526fce;text-align:right;font-size:1.6rem;}
.panel-title {text-align:center;color:#71362f;font-size:1.55rem;font-weight:700;margin:.25rem 0 .55rem;}
.prediction-line {text-align:center;color:#71362f;font-size:1.15rem;margin:0 0 1.1rem;}
.prob-list {max-height:520px;overflow-y:auto;padding:4px 8px 4px 0;}
.prob-row {display:grid;grid-template-columns:minmax(110px,145px) minmax(80px,1fr) 54px;gap:10px;align-items:center;margin:5px 0;font-size:.83rem;}
.prob-label {color:#30343b;text-align:right;}.prob-track {height:15px;background:#f0f1f3;border:1px solid #dedfe2;}
.prob-fill {height:100%;background:#ec1717;min-width:1px;}.prob-value {color:#474747;font-variant-numeric:tabular-nums;}
.breed-copy {color:#71362f;line-height:1.8;font-size:1rem;}div[data-testid="stFileUploader"] section {border-color:#61b7cf;}
div[data-testid="stImage"] {display:flex;justify-content:center;}
@media(max-width:800px){.topbar{grid-template-columns:1fr;text-align:center;}.paw{display:none;}.page-title{font-size:1.3rem;}.prob-row{grid-template-columns:minmax(90px,120px) minmax(50px,1fr) 48px;gap:6px;}}
</style>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="topbar"><div class="brand">🐾 PET VISION<small>Phân loại ảnh thú cưng</small></div>'
    '<div class="page-title">Phân loại giống chó mèo</div><div class="paw">🐶 🐱</div></div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_classifier(Path("artifacts"))


left, right = st.columns([0.9, 2.1], gap="medium")
image = predictions = metadata = None
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Ảnh đầu vào</div>', unsafe_allow_html=True)
        preview = st.empty()
        uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "webp"])
        camera_photo = st.camera_input("Mở webcam / chụp ảnh")
        source = camera_photo or uploaded
        if source:
            image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            preview.image(image, caption="Ảnh đã chọn", width=280)
        else:
            preview.info("Tải ảnh lên hoặc mở webcam để xem kết quả.")

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Kết quả</div>', unsafe_allow_html=True)
        if image is None:
            st.info("Biểu đồ xác suất sẽ hiển thị sau khi bạn chọn ảnh.")
        else:
            try:
                model, metadata, device, transform = get_model()
                predictions = predict_top_k(model, metadata, device, transform, image, len(metadata["class_names"]))
                best = predictions[0]
                st.markdown(
                    f'<div class="prediction-line">Dự đoán {best["confidence"]:.2%} là giống '
                    f'{escape(best["species"])} <b>{escape(best["breed"])}</b></div>',
                    unsafe_allow_html=True,
                )
                rows = []
                for item in predictions:
                    value = item["confidence"]
                    width = max(0.5, value * 100)
                    rows.append(
                        '<div class="prob-row">'
                        f'<div class="prob-label">{escape(item["breed"])}</div>'
                        f'<div class="prob-track"><div class="prob-fill" style="width:{width:.3f}%"></div></div>'
                        f'<div class="prob-value">{value:.2%}</div></div>'
                    )
                st.markdown('<div class="prob-list">' + "".join(rows) + "</div>", unsafe_allow_html=True)
            except FileNotFoundError as error:
                st.error(str(error))

if predictions:
    best = predictions[0]
    with st.container(border=True):
        st.markdown(
            f'<div class="panel-title">Thông tin về giống {escape(best["species"])} {escape(best["breed"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="breed-copy">'
            f'Mô hình dự đoán ảnh thuộc giống <b>{escape(best["breed"])}</b> '
            f'({escape(best["species"])}), với độ tin cậy {best["confidence"]:.2%}. '
            f'Biểu đồ hiển thị xác suất của toàn bộ {len(metadata["class_names"])} giống. '
            'Kết quả có thể thay đổi theo góc chụp, ánh sáng và chất lượng ảnh; không dùng thay cho xác nhận phả hệ.'
            '</div>',
            unsafe_allow_html=True,
        )
