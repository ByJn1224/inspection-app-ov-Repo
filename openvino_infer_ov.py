"""
ver2.py  ─  ver1.py + 웹캠 입력 추가
════════════════════════════════════════════════════════════════════
ver1.py의 로직(모델 로드 / 전처리 / 추론)은 그대로 두고,
입력 방식만 "파일 업로드" / "웹캠" 두 가지로 늘렸다.

[추가된 것]
  - st.radio로 입력 방식 선택
  - st.camera_input() → 웹캠 촬영 (브라우저가 카메라 권한을 요청함)
    ※ infer_keras.py의 OpenCV(cv2.VideoCapture/imshow) 방식은
      로컬 GUI 창을 띄우는 방식이라 브라우저 기반인 Streamlit에서는
      동작하지 않는다. 그래서 웹캠 부분은 st.camera_input()으로
      새로 작성했고, 그 이후(전처리~추론~결과표시)는 완전히 동일하다.

[실행 방법]
  pip install streamlit openvino pillow numpy
  streamlit run ver2.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import openvino as ov ##오픈비노 라이브러리 임포트

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_PATH = "./model/leather_model.xml"   ## IR 경로
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]


# ─────────────────────────────────────────────────────────────────
# 1. 모델 로드 (캐싱)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"모델 파일이 없습니다: {MODEL_PATH}")
        st.stop()
    return ov.Core().compile_model(MODEL_PATH, "CPU") ##


# ─────────────────────────────────────────────────────────────────
# 2. 이미지 전처리
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)  ## 변경 (RGB→BGR, 평균 차감)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# 3. 추론
# ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr   = preprocess(pil_img)
    prob = float(model(arr)[0].ravel()[0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────────────────────────
# 4. 이미지 입력 : 파일 업로드 / 웹캠 (둘 다 PIL 이미지로 통일)
#    file_uploader와 camera_input 둘 다 파일과 유사한 객체를 반환하므로
#    Image.open()으로 여는 방식이 동일하게 적용된다.
# ─────────────────────────────────────────────────────────────────
def get_input_image():
    input_mode = st.radio("입력 방식", ["파일 업로드", "웹캠"], horizontal=True)

    if input_mode == "파일 업로드":
        raw = st.file_uploader("검사할 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    else:
        raw = st.camera_input("카메라로 촬영하세요")

    if raw is None:
        return None

    try:
        return Image.open(raw).convert("RGB")
    except UnidentifiedImageError:
        st.error("이미지를 읽을 수 없습니다. 다시 시도해주세요.")
        return None


# ─────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="가죽 표면 결함 검사", page_icon="🔍")
    st.title("🔍 가죽 표면 결함 검사")
    st.caption(".keras 모델 기반 정상 / 불량 이미지 분류")

    model = load_model()

    pil_img = get_input_image()
    if pil_img is None:
        return

    st.image(pil_img, caption="입력 이미지", use_container_width=True)

    if st.button("추론 실행", type="primary"):
        with st.spinner("추론 중..."):
            label, prob = predict(model, pil_img)

        normal_prob, defect_prob = 1 - prob, prob

        if label == "불량":
            st.error(f"예측 결과 : **{label}**")
        else:
            st.success(f"예측 결과 : **{label}**")

        col1, col2 = st.columns(2)
        col1.metric("정상 확률", f"{normal_prob:.1%}")
        col2.metric("불량 확률", f"{defect_prob:.1%}")

        st.subheader("클래스별 확률")
        chart_df = pd.DataFrame(
            {"확률": [normal_prob, defect_prob]},
            index=["정상", "불량"],
        )
        st.bar_chart(chart_df, y="확률", height=240)


if __name__ == "__main__":
    main()
