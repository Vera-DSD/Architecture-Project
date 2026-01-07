import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import matplotlib.pyplot as plt


# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="Архитектурный поиск",
    page_icon="🏛️",
    layout="wide"
)

EMBEDDINGS_FILE = 'clip_embeddings.npy'
PATHS_FILE = 'clip_image_paths.npy'
DATASET_PATH = 'pexels_architecture_fixed'
MODEL_NAME = 'ViT-B/32' 
TOP_K = 5

# ======== Загрузка модели и даннных ========
@st.cache_resource
def load_clip_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess, device

@st.cache_data
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error(f'Файл {EMBEDDINGS_FILE} не найден !')
        st.info('Сначала запустите создание базы данных')
        return None, None
    
    embeddings = np.load(EMBEDDINGS_FILE)
    paths = np.load(PATHS_FILE, allow_pickle=True)
    return embeddings, paths


# ========== Создает эмбеддинг==========

def get_image_embedding(_model, _preprocess, _device, img_path_or_uploaded_fiile):
    """Создает эмбеддинг изображения"""
    try:
        if hasattr(img_path_or_uploaded_fiile, 'read'):
            image = Image.open(img_path_or_uploaded_fiile).convert('RGB')
        else:
            image = Image.open(img_path_or_uploaded_fiile).convert('RGB')

        image_input = _preprocess(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            embedding = _model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        st.error(f'Ошибка обработки изображений: {e}')
        return None
    
def get_text_embedding(_model, _device, text_query):
    """Создает эмбеддинг текста"""
    try:
        text_input = clip.tokenize([f'a photo of {text_query}']).to(_device)
        with torch.no_grad():
            embedding = _model.encode_text(text_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        st.error('Ошибка обработки текста: {e}')
        return None
    
# ============== Поиск похожих ================  
    
def search_by_image(query_img, _model, _preprocess, _device, embedding, paths, top_k=TOP_K):
    """Поиск похожих изображений"""
    query_emb = get_image_embedding(_model, _preprocess, _device, query_img)
    if query_emb is None:
        return []
    
    query_emb_2d = query_emb.reshape(1, -1)
    similarities = 1 - cdist(query_emb_2d, embedding, 'cosine')[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(paths[idx], similarities[idx]) for idx in top_indices]

def search_by_text(query_text, _model, _device, embeddings, paths, top_k = TOP_K):
    """Поиск изображений по тексту"""
    query_emb = get_text_embedding(_model, _device, query_text)
    if query_emb is None:
        return []
    
    query_emb_2d = query_emb.reshape(1,-1)
    similarities = (embeddings @ query_emb_2d.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(paths[idx], similarities[idx]) for idx in top_indices]

#======================================================================

def zero_shot_classify(query_img, _model, _preprocess, _device, class_descriptions=None):
    """Классификация без обучения"""
    if class_descriptions is None:
        class_descriptions =[
            'modern skyscraper with glass fasade',
            'classical building with columns',
            'gothic cathedral with stained glass',
            'traditional wooden house',
            'brutalist concrete structure',
            'industrial warehouse with brick walls',
            'contemporary minimalist building',
            'art deco skyscraper',
            'medieval castle with towers',
            'modernist villa with clean lines'
        ]
    
    image_emb = get_image_embedding(_model, _preprocess, _device, query_img)
    if image_emb is None:
        return []
    
    # Токенизация текста
    text_inputs = clip.tokenize(class_descriptions).to(_device)

    with torch.no_grad():
        text_embeddings = _model.encode_text(text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

# Вычисляем сходство

        image_emb_2d = image_emb.reshape(1, -1)
        similarity = (100.0 * image_emb_2d @ text_embeddings.T)
        probs = similarity.softmax(dim=-1)

        values, indices = probs[0].topk(min(5, len(class_descriptions)))

    return [(class_descriptions[idx], val.item()) for val, idx in zip(values, indices)]


# ========== ФУНКЦИЯ ДЛЯ СОЗДАНИЯ БАЗЫ ДАННЫХ ==========

def create_database(_model, _preprocess, _device):
    """Создание базы эмбеддингов (запускается вручную)"""

    st.info('Создание базы данных... Это может занять несколько минут.')
    all_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        all_paths.extend(Path(DATASET_PATH).rglob(ext))

    st.write(f'Найдено {len(all_paths)} изображений')

    embeddings = []
    valid_paths = []
    progress_bar = st.progress(0)

    for i, img_path in enumerate(all_paths):
        emb = get_image_embedding(_model, _preprocess, _device, str(img_path))
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(str(img_path))

            if (i+1) % 10 == 0 or (i+1) == len(all_paths):
                progress_bar.progress((i+1)/ len(all_paths))
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            np.save(EMBEDDINGS_FILE, embeddings_array)
            np.save(PATHS_FILE, np.array(valid_paths))

            st.success(f'База создана: {embeddings_array.shape[0]} эмбеддингов')
            st.write(f'Размерность: {embeddings_array, np.array(valid_paths)}')
            return embeddings_array, np.array(valid_paths)
        else:
            st.error('Не удалось создать эмбеддинги')
            return None, None
        
 # ========== ОСНОВНОЙ ИНТЕРФЕЙС ==========

def main():
    st.title("🏛️ Архитектурный поиск с использованием CLIP")
    st.markdown("Система поиска и классификации архитектурных изображений")

    model, preprocess, device = load_clip_model()
    embeddings, paths = load_embeddings()

    with st.sidebar:
        st.header('⚙️ Настройки')


        mode = st.radio(
            "Выберите режим:",
            ["🔍 Поиск по тексту", "🖼️ Поиск по изображению", "🏷️ Классификация стиля", "🗃️ Создать базу данных"]
        )

        top_k = st.slider("Количество результатов:", 1, 20, TOP_K)

        st.divider()
        st.header("ℹ️ Информация")
        if embeddings is not None:
            st.write(f"**База данных:** {len(paths)} изображений")
            st.write(f"**Размерность эмбеддингов:** {embeddings.shape[1]}")
        st.write(f"**Устройство:** {device}")
        st.write(f"**Модель:** CLIP ViT-B/32")

    # Основное содержимое
    if mode == "🔍 Поиск по тексту":
        st.header("Поиск изображений по текстовому описанию")
        
        # Примеры запросов
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("modern glass skyscraper"):
                st.session_state.text_query = "modern glass skyscraper"
        with col2:
            if st.button("classical building with columns"):
                st.session_state.text_query = "classical building with columns"
        with col3:
            if st.button("gothic cathedral"):
                st.session_state.text_query = "gothic cathedral"
        
        # Поле для ввода текста
        text_query = st.text_input(
            "Введите описание архитектуры:",
            value=getattr(st.session_state, 'text_query', 'modern building'),
            placeholder="например: modern glass skyscraper"
        )
        
        if st.button("Найти", type="primary") or text_query:
            if embeddings is None:
                st.error("База данных не загружена!")
                return
            
            with st.spinner("Ищем похожие изображения..."):
                results = search_by_text(text_query, model, device, embeddings, paths, top_k)
            
            if results:
                st.success(f"Найдено {len(results)} изображений")
                
                # Отображение результатов в сетке
                cols = st.columns(min(4, len(results)))
                for idx, (col, (path, score)) in enumerate(zip(cols, results)):
                    with col:
                        try:
                            img = Image.open(path).convert('RGB')
                            st.image(img, use_container_width=True)
                            st.caption(f"**Сходство:** {score:.3f}")
                            st.caption(f"**Файл:** {os.path.basename(path)}")
                            
                            # Кнопка для детального просмотра
                            if st.button(f"🔍 Подробнее {idx+1}", key=f"detail_{idx}"):
                                st.session_state.selected_image = path
                        except Exception as e:
                            st.error(f"Ошибка загрузки: {e}")
                
                # Детальный просмотр выбранного изображения
                if hasattr(st.session_state, 'selected_image'):
                    st.divider()
                    st.subheader("📸 Детальный просмотр")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(st.session_state.selected_image, use_container_width=True)
                    with col2:
                        st.write(f"**Полный путь:** {st.session_state.selected_image}")
                        st.write(f"**Размер:** {Image.open(st.session_state.selected_image).size}")
            else:
                st.warning("Не найдено результатов")
    
    elif mode == "🖼️ Поиск по изображению":
        st.header("Поиск похожих изображений")
        
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Загрузите изображение для поиска",
            type=['jpg', 'jpeg', 'png'],
            help="Загрузите изображение архитектуры"
        )
        
        # Примеры изображений
        st.write("Или выберите пример из базы:")
        if paths is not None and len(paths) > 0:
            sample_cols = st.columns(4)
            sample_indices = np.random.choice(len(paths), 4, replace=False)
            for col, idx in zip(sample_cols, sample_indices):
                with col:
                    if st.button(f"Пример {idx+1}", key=f"sample_{idx}"):
                        st.session_state.sample_image = paths[idx]
        
        # Используем загруженное или выбранное изображение
        query_image = uploaded_file if uploaded_file else getattr(st.session_state, 'sample_image', None)
        
        if query_image:
            # Показываем запрос
            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(query_image, 'read'):
                    img = Image.open(query_image).convert('RGB')
                    st.image(img, caption="Ваш запрос", use_container_width=True)
                else:
                    st.image(query_image, caption="Ваш запрос", use_container_width=True)
            
            with col2:
                st.write("**Информация о запросе:**")
                if hasattr(query_image, 'name'):
                    st.write(f"**Имя файла:** {query_image.name}")
                    st.write(f"**Тип:** {query_image.type}")
                else:
                    st.write(f"**Путь:** {query_image}")
            
            if st.button("Найти похожие", type="primary") and embeddings is not None:
                with st.spinner("Ищем похожие изображения..."):
                    results = search_by_image(query_image, model, preprocess, device, embeddings, paths, top_k)
                
                if results:
                    st.subheader(f"🎯 Топ-{len(results)} похожих изображений:")
                    
                    # Сетка результатов
                    for i in range(0, len(results), 4):
                        cols = st.columns(4)
                        for col_idx in range(4):
                            if i + col_idx < len(results):
                                path, score = results[i + col_idx]
                                with cols[col_idx]:
                                    try:
                                        img = Image.open(path).convert('RGB')
                                        st.image(img, use_container_width=True)
                                        
                                        # Прогресс-бар для наглядности сходства
                                        st.progress(float(score))
                                        
                                        st.caption(f"**Сходство:** {score:.3f}")
                                        st.caption(f"**{os.path.basename(path)}**")
                                    except Exception as e:
                                        st.error(f"Ошибка: {e}")
    
    elif mode == "🏷️ Классификация стиля":
        st.header("Определение архитектурного стиля")
        
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Загрузите изображение для классификации",
            type=['jpg', 'jpeg', 'png']
        )
        
        # Настройка классов
        st.subheader("Настройка классов для классификации")
        
        default_classes = [
            "modern skyscraper with glass facade",
            "classical building with columns",
            "gothic cathedral with stained glass",
            "traditional wooden house",
            "brutalist concrete structure"
        ]
        
        custom_classes = st.text_area(
            "Введите описания классов (каждое с новой строки):",
            value="\n".join(default_classes),
            height=150,
            help="Каждая строка - отдельный класс. Например: 'modern glass building'"
        )
        
        class_list = [c.strip() for c in custom_classes.split('\n') if c.strip()]
        
        if uploaded_file and class_list:
            # Показываем изображение
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption="Изображение для классификации", use_container_width=True)
            
            if st.button("Классифицировать", type="primary"):
                with st.spinner("Анализируем стиль..."):
                    results = zero_shot_classify(uploaded_file, model, preprocess, device, class_list)
                
                if results:
                    st.subheader("📊 Результаты классификации:")
                    
                    # Столбчатая диаграмма
                    classes = [r[0] for r in results]
                    scores = [r[1] for r in results]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(classes, scores, color='skyblue')
                    ax.set_xlabel('Вероятность')
                    ax.set_title('Вероятности архитектурных стилей')
                    
                    # Добавляем значения на график
                    for bar, score in zip(bars, scores):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{score:.2%}', va='center')
                    
                    st.pyplot(fig)
                    
                    # Таблица с результатами
                    st.subheader("📋 Детальные результаты:")
                    for i, (class_name, prob) in enumerate(results, 1):
                        st.write(f"{i}. **{class_name}** → {prob:.2%}")
                        
                        # Прогресс-бар для каждого класса
                        st.progress(float(prob))
    
    elif mode == "🗃️ Создать базы данных":
        st.header("Создание/обновление базы данных")
        
        st.warning("⚠️ Внимание: Это действие перезапишет существующую базу данных!")
        
        if st.button("Создать базу данных", type="primary"):
            if not os.path.exists(DATASET_PATH):
                st.error(f"Папка {DATASET_PATH} не найдена!")
                return
            
            embeddings_new, paths_new = create_database(model, preprocess, device)
            if embeddings_new is not None:
                st.success("База данных успешно создана!")
                st.rerun()  

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    # Проверка зависимостей
    try:
        import streamlit
        import torch
        import clip
        main()
    except ImportError as e:
        st.error(f"Не удалось импортировать модуль: {e}")
       


