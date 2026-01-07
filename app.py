import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import os

# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(page_title="Архитектурный поиск", layout="wide")

EMBEDDINGS_FILE = 'clip_embeddings.npy'  # Исправлено!
PATHS_FILE = 'clip_image_paths.npy'      # Исправлено!
TOP_K = 5

# ========== ЗАГРУЗКА ==========
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

@st.cache_data
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error(f"Файл {EMBEDDINGS_FILE} не найден!")
        return None, None
    return np.load(EMBEDDINGS_FILE), np.load(PATHS_FILE, allow_pickle=True)

# ========== ПОИСК ==========
def search_by_text(query, model, device, embeddings, paths, top_k=5):
    text_input = clip.tokenize([f"a photo of {query}"]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_input)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    
    similarities = (embeddings @ text_emb.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(paths[i], similarities[i]) for i in top_indices]

def main():
    st.title("🏛️ Архитектурный поиск")
    
    # Загрузка модели
    model, preprocess, device = load_model()
    
    # Режимы
    mode = st.sidebar.radio("Режим:", ["🔍 Поиск по тексту", "🖼️ Поиск по изображению"])
    top_k = st.sidebar.slider("Количество результатов:", 1, 10, 5)
    
    if mode == "🔍 Поиск по тексту":
        query = st.text_input("Опишите архитектуру:", "modern building")
        if st.button("Найти"):
            embeddings, paths = load_embeddings()
            if embeddings is None:
                st.error("База данных не найдена!")
                st.info("Сначала создайте базу данных с изображениями")
                return
            
            results = search_by_text(query, model, device, embeddings, paths, top_k)
            cols = st.columns(top_k)
            for col, (path, score) in zip(cols, results):
                with col:
                    img = Image.open(path).convert('RGB')
                    st.image(img, use_container_width=True)
                    st.caption(f"Сходство: {score:.3f}")
    else:
        st.write("Режим поиска по изображению будет добавлен позже")

if __name__ == "__main__":
    main()
