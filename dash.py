%%writefile dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ====================== КОНФИГУРАЦИЯ ======================
# Меняй только эту часть под свой проект!

# Название проекта
PROJECT_NAME = "Портал Бессмертия"

# Путь к файлам (относительно места запуска dashboard.py)
TRAIN_FILE = 'train_minmax.csv'      # основной датасет (для визуализации)
TEST_FILE = 'test_minmax.csv'        # тестовый (не обязателен)
MODEL_FILE = 'best_model.pkl'         # обученная модель
FEATURE_NAMES_FILE = 'feature_names.pkl'  # список признаков (list[str])
METRICS_FILE = 'model_metrics.csv'   # таблица с метриками всех моделей (опционально)

# Название целевой переменной
TARGET_COLUMN = 'Гармония Бессмертия'

# Порог стабильности (для gauge)
STABILITY_THRESHOLD = 0.95

# Цвета для gauge
GAUGE_COLORS = {
    'low': '#ffcccc',    # < 0.85
    'medium': '#ffffcc', # 0.85–0.95
    'high': '#ccffcc'    # > 0.95
}

# =========================================================

# Настройка страницы
st.set_page_config(
    page_title=f"{PROJECT_NAME} - Анализ данных",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"🔮 Дашборд анализа: {PROJECT_NAME}")
st.markdown("---")

# Боковая панель
st.sidebar.header("Навигация")
page = st.sidebar.selectbox(
    "Выберите раздел",
    ["Обзор данных", "Визуализация признаков", "Корреляционный анализ", 
     "Результаты моделей", "Предсказания"]
)

# ====================== ЗАГРУЗКА ДАННЫХ ======================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE) if os.path.exists(TEST_FILE) else None
        return train, test
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(), None

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE) if os.path.exists(FEATURE_NAMES_FILE) else None
        st.sidebar.success("✅ Модель загружена")
        return model, feature_names
    except Exception as e:
        st.sidebar.warning(f"⚠️ Модель не найдена: {e}")
        return None, None

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv(METRICS_FILE)
    except:
        return None

train_data, test_data = load_data()
model, feature_names = load_model()
metrics_df = load_metrics()

# ====================== СТРАНИЦЫ ======================

if page == "Обзор данных":
    st.header("📊 Обзор данных")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Обучающая выборка", len(train_data))
    with col2:
        st.metric("Тестовая выборка", len(test_data) if test_data is not None else "—")
    with col3:
        st.metric("Признаков", train_data.shape[1] - 1)
    with col4:
        st.metric("Пропусков", train_data.isnull().sum().sum())
    
    st.subheader("Первые строки")
    st.dataframe(train_data.head(10), use_container_width=True)
    
    st.subheader("Описательная статистика")
    st.dataframe(train_data.describe().round(4), use_container_width=True)

elif page == "Визуализация признаков":
    st.header("📈 Визуализация признаков")
    
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)
    
    selected_feature = st.selectbox("Выберите признак", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(train_data, x=selected_feature, nbins=50, title=f"Гистограмма: {selected_feature}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(train_data, y=selected_feature, title=f"Boxplot: {selected_feature}")
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Статистика
    stats = train_data[selected_feature].describe()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Среднее", f"{stats['mean']:.4f}")
    with col2: st.metric("Медиана", f"{stats['50%']:.4f}")
    with col3: st.metric("Стд. откл.", f"{stats['std']:.4f}")
    with col4: st.metric("Мин", f"{stats['min']:.4f}")
    with col5: st.metric("Макс", f"{stats['max']:.4f}")

elif page == "Корреляционный анализ":
    st.header("🔗 Корреляционный анализ")
    
    numeric_data = train_data.select_dtypes(include=[np.number])
    
    st.subheader("Тепловая карта корреляций")
    fig_size = st.slider("Размер карты", 8, 20, 12)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)
    plt.title('Корреляционная матрица')
    st.pyplot(fig)
    plt.close(fig)
    
    if TARGET_COLUMN in numeric_data.columns:
        st.subheader("Корреляция с целевой переменной")
        target_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False)
        
        fig_bar = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                         title='Абсолютная корреляция с целевой',
                         labels={'x': 'Корреляция', 'y': 'Признак'})
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Результаты моделей":
    st.header("🤖 Результаты моделей")
    
    if metrics_df is not None:
        st.dataframe(metrics_df.style.highlight_min(subset=['RMSE_test'], color='lightgreen'), use_container_width=True)
        
        best = metrics_df.loc[metrics_df['RMSE_test'].idxmin()]
        st.success(f"🏆 Лучшая модель: **{best['Model']}** на датасете **{best['Dataset']}**")
        st.info(f"RMSE_test = {best['RMSE_test']:.6f} | R²_test = {best['R2_test']:.4f}")
        
        # Графики сравнения
        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = px.bar(metrics_df, x='Model', y='RMSE_test', color='Dataset', title='RMSE на тесте', barmode='group')
            st.plotly_chart(fig_rmse, use_container_width=True)
        with col2:
            fig_r2 = px.bar(metrics_df, x='Model', y='R2_test', color='Dataset', title='R² на тесте', barmode='group')
            st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.warning("Метрики не загружены — используйте демо-режим")

elif page == "Предсказания":
    st.header("🔮 Интерактивное предсказание")
    
    if model is None:
        st.error("Модель не загружена")
        st.stop()
    
    st.success(f"Модель: {type(model).__name__}")
    
    input_data = {}
    if feature_names:
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                default = float(train_data[feat].mean()) if feat in train_data.columns else 0.0
                input_data[feat] = st.number_input(feat, value=default, format="%.6f")
    else:
        st.warning("Список признаков не загружен — ввод вручную")
        manual_features = st.text_input("Признаки через запятую")
        if manual_features:
            for feat in manual_features.split(','):
                input_data[feat.strip()] = st.number_input(feat.strip(), value=0.0)
    
    if st.button("Предсказать", type="primary"):
        input_df = pd.DataFrame([input_data])
        
        # Проверяем порядок колонок
        if feature_names:
            input_df = input_df[feature_names]
        
        pred = model.predict(input_df)[0]
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': TARGET_COLUMN},
            delta = {'reference': STABILITY_THRESHOLD},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.85], 'color': GAUGE_COLORS['low']},
                    {'range': [0.85, STABILITY_THRESHOLD], 'color': GAUGE_COLORS['medium']},
                    {'range': [STABILITY_THRESHOLD, 1], 'color': GAUGE_COLORS['high']}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': STABILITY_THRESHOLD}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Интерпретация
        if pred >= STABILITY_THRESHOLD:
            st.success(f"**{pred:.6f}** — Система стабильна!")
        elif pred >= 0.85:
            st.warning(f"**{pred:.6f}** — Требуется внимание")
        else:
            st.error(f"**{pred:.6f}** — Критическое состояние!")

# Футер
st.markdown("---")
st.caption(f"Дашборд для проекта {PROJECT_NAME} | 2026")
