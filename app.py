import io
import os

import streamlit as st
from PIL import Image

from pipeline import ImageTooLargeError, process_image
from postprocessing.preview import apply_postprocessing

BASE_PREVIEW_WIDTH = 200
DEMO_DIR = "demo_images"


def reset_result():
    st.session_state.output_image = None
    st.session_state.base_output_image = None
    st.session_state.preview_image = None
    st.session_state.blur_strength = 0.0
    st.session_state.sharpen_strength = 0


def update_preview():
    base = st.session_state.base_output_image
    if base is None:
        st.session_state.preview_image = None
        return

    st.session_state.preview_image = apply_postprocessing(
        base,
        blur_strength=st.session_state.blur_strength,
        sharpen_strength=st.session_state.sharpen_strength,
    )


def reset_postprocessing():
    st.session_state.blur_strength = 0.0
    st.session_state.sharpen_strength = 0
    update_preview()


st.set_page_config(
    page_title="Восстановление старых фотографий",
    layout="centered",
)


def get_demo_images():
    if not os.path.exists(DEMO_DIR):
        return []

    files = os.listdir(DEMO_DIR)

    images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    return sorted(images)


st.title("Восстановление старых фотографий")
st.markdown("Сервис для улучшения качества старых и повреждённых изображений.")

# -----------------------------
# Session state
# -----------------------------
if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "output_image" not in st.session_state:
    st.session_state.output_image = None

if "base_output_image" not in st.session_state:
    st.session_state.base_output_image = None

if "preview_image" not in st.session_state:
    st.session_state.preview_image = None

if "blur_strength" not in st.session_state:
    st.session_state.blur_strength = 0.0

if "sharpen_strength" not in st.session_state:
    st.session_state.sharpen_strength = 0


# -----------------------------
# Источник изображения
# -----------------------------
source_mode = st.radio(
    "Источник изображения",
    ["Загрузить своё", "Использовать пример"],
    horizontal=True,
    on_change=reset_result,
)

# -----------------------------
# Загрузка / выбор изображения
# -----------------------------
if source_mode == "Загрузить своё":
    uploaded = st.file_uploader(
        "Загрузите изображение",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.stop()

    input_image = Image.open(uploaded)
    current_image_id = hash(uploaded.getvalue())

else:
    demo_images = get_demo_images()

    if not demo_images:
        st.info("В хранилище пока нет изображений.")
        st.stop()

    demo_choice = st.selectbox(
        "Выберите изображение",
        demo_images,
        on_change=reset_result,
    )

    input_image = Image.open(os.path.join(DEMO_DIR, demo_choice))
    current_image_id = demo_choice

# новая картинка → сбрасываем результат
if st.session_state.image_id != current_image_id:
    st.session_state.image_id = current_image_id
    reset_result()


st.subheader("Исходное изображение")
st.image(input_image, width=BASE_PREVIEW_WIDTH)

st.divider()

# -----------------------------
# Выбор режима — ВСЕГДА ВИДЕН
# -----------------------------
ui_mode = st.radio(
    "Режим работы",
    ["Простой", "Продвинутый"],
    horizontal=True,
    on_change=reset_result,
)

st.divider()

# -----------------------------
# ПРОСТОЙ РЕЖИМ
# -----------------------------
if ui_mode == "Простой":
    st.subheader("Параметры обработки")

    upscale_mode = st.selectbox(
        "Увеличение изображения",
        [
            "Без изменения размера (повышение качества)",
            "Увеличить в 2 раза",
            "Увеличить в 4 раза",
        ],
        on_change=reset_result,
    )

    do_colorize = st.checkbox(
        "Раскрашивать изображение",
        value=True,
        on_change=reset_result,
    )

    if upscale_mode.startswith("Без"):
        preview_width = BASE_PREVIEW_WIDTH
        pipeline_upscale_mode = "enhance"
        pipeline_upscale_scale = 2
    elif "2" in upscale_mode:
        preview_width = BASE_PREVIEW_WIDTH * 2
        pipeline_upscale_mode = "resize"
        pipeline_upscale_scale = 2
    else:
        preview_width = BASE_PREVIEW_WIDTH * 4
        pipeline_upscale_mode = "resize"
        pipeline_upscale_scale = 4

    run = st.button("Восстановить")

    if run:
        with st.spinner("Обрабатываем изображение..."):
            try:
                base = process_image(
                    input_image,
                    do_colorize=do_colorize,
                    upscale_mode=pipeline_upscale_mode,
                    upscale_scale=pipeline_upscale_scale,
                )

            except ImageTooLargeError as e:
                st.error(str(e))
                st.stop()

            st.session_state.base_output_image = base
            st.session_state.preview_image = base


# -----------------------------
# ПРОДВИНУТЫЙ РЕЖИМ
# -----------------------------
else:
    st.subheader("Этапы обработки")

    col1, col2 = st.columns(2)

    with col1:
        do_deblur = st.checkbox(
            "Устранение размытия",
            value=True,
            on_change=reset_result,
        )
        do_colorize = st.checkbox(
            "Раскрашивание",
            value=True,
            on_change=reset_result,
        )

    with col2:
        do_upscale = st.checkbox(
            "Увеличение изображения",
            value=True,
            on_change=reset_result,
        )
        do_post_denoise = st.checkbox(
            "Стабилизация после увеличения",
            value=True,
            on_change=reset_result,
        )
        do_sharpen = st.checkbox(
            "Повышение резкости",
            value=True,
            on_change=reset_result,
        )

    preview_width = BASE_PREVIEW_WIDTH

    if not do_upscale:
        pipeline_upscale_mode = None
        pipeline_upscale_scale = 2

    else:
        upscale_mode = st.selectbox(
            "Режим увеличения",
            [
                "Без изменения размера",
                "Увеличить в 2 раза",
                "Увеличить в 4 раза",
            ],
            on_change=reset_result,
        )

        if upscale_mode.startswith("Без"):
            preview_width = BASE_PREVIEW_WIDTH
            pipeline_upscale_mode = "enhance"
            pipeline_upscale_scale = 2
        elif "2" in upscale_mode:
            preview_width = BASE_PREVIEW_WIDTH * 2
            pipeline_upscale_mode = "resize"
            pipeline_upscale_scale = 2
        else:
            preview_width = BASE_PREVIEW_WIDTH * 4
            pipeline_upscale_mode = "resize"
            pipeline_upscale_scale = 4

    run = st.button("Обработать")

    if run:
        with st.spinner("Обрабатываем изображение..."):
            try:
                base = process_image(
                    input_image,
                    do_deblur=do_deblur,
                    do_colorize=do_colorize,
                    upscale_mode=pipeline_upscale_mode,
                    upscale_scale=pipeline_upscale_scale,
                    do_post_denoise=do_post_denoise,
                    do_sharpen=do_sharpen,
                )

            except ImageTooLargeError as e:
                st.error(str(e))
                st.stop()

            st.session_state.base_output_image = base
            st.session_state.preview_image = base


# -----------------------------
# Постобработка (онлайн) — после любого режима
# -----------------------------
if st.session_state.base_output_image is not None:
    st.divider()
    st.subheader("Финальная настройка")

    st.slider(
        "Размытие (смягчить изображение)",
        min_value=0.0,
        max_value=3.0,
        step=0.1,
        key="blur_strength",
        on_change=update_preview,
    )

    st.slider(
        "Резкость",
        min_value=0,
        max_value=100,
        step=1,
        key="sharpen_strength",
        on_change=update_preview,
    )

    st.button(
        "Сбросить постобработку",
        on_click=reset_postprocessing,
    )


# -----------------------------
# Результат + скачивание
# -----------------------------
if st.session_state.preview_image is not None:
    with st.container():
        st.subheader("Результат")

        st.image(
            st.session_state.preview_image,
            width=preview_width,
        )

        buf = io.BytesIO()
        st.session_state.preview_image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            "Скачать результат",
            data=buf,
            file_name="result.png",
            mime="image/png",
        )
