import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Настройка страницы
st.set_page_config(
    page_title="Портал Бессмертия - Анализ данных",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🔮 Дашборд анализа Портала Бессмертия")
st.markdown("---")

# Боковая панель
st.sidebar.header("Настройки")
page = st.sidebar.selectbox(
    "Выберите раздел",
    ["Обзор данных", "Визуализация признаков", "Корреляционный анализ", 
     "Результаты моделей", "Предсказания"]
)

# Загрузка данных
@st.cache_data
def load_data():
    try:
        train = pd.read_csv('train_minmax.csv')
        test = pd.read_csv('test_minmax.csv')
        return train, test
    except Exception as e:
        st.warning(f"Не удалось загрузить данные: {e}")
        train = pd.DataFrame({
            'Признак_1': np.random.uniform(0, 1, 100),
            'Признак_2': np.random.uniform(0, 1, 100),
            'Гармония Бессмертия': np.random.uniform(0.8, 1.0, 100)
        })
        test = train.copy()
        return train, test

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        st.sidebar.success("✅ Модель загружена")
        return model, feature_names
    except Exception as e:
        st.sidebar.warning(f"⚠️ Модель не загружена: {e}")
        return None, None

# Загрузка метрик
@st.cache_data
def load_metrics():
    try:
        return pd.read_csv('model_metrics.csv')
    except:
        return None

train_data, test_data = load_data()
model, feature_names = load_model()
metrics_df = load_metrics()

# СТРАНИЦА 1: Обзор данных
if page == "Обзор данных":
    st.header("📊 Обзор данных")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Обучающая выборка", train_data.shape[0])
    with col2:
        st.metric("Тестовая выборка", test_data.shape[0])
    with col3:
        st.metric("Количество признаков", train_data.shape[1] - 1)
    with col4:
        st.metric("Пропусков (train)", train_data.isnull().sum().sum())
    
    st.subheader("Первые строки данных")
    st.dataframe(train_data.head(10), use_container_width=True)
    
    st.subheader("Статистическое описание")
    st.dataframe(train_data.describe(), use_container_width=True)

# СТРАНИЦА 2: Визуализация признаков
elif page == "Визуализация признаков":
    st.header("📈 Визуализация признаков")
    
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    
    st.subheader("Распределение признаков")
    selected_feature = st.selectbox("Выберите признак", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        train_data[selected_feature].hist(bins=50, ax=ax1, edgecolor='black')
        ax1.set_title(f'Гистограмма: {selected_feature}')
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel('Частота')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        train_data.boxplot(column=selected_feature, ax=ax2)
        ax2.set_title(f'Boxplot: {selected_feature}')
        ax2.set_ylabel(selected_feature)
        st.pyplot(fig2)
        plt.close()
    
    st.subheader("Статистика выбранного признака")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Среднее", f"{train_data[selected_feature].mean():.4f}")
    with col2:
        st.metric("Медиана", f"{train_data[selected_feature].median():.4f}")
    with col3:
        st.metric("Ст. откл.", f"{train_data[selected_feature].std():.4f}")
    with col4:
        st.metric("Мин", f"{train_data[selected_feature].min():.4f}")
    with col5:
        st.metric("Макс", f"{train_data[selected_feature].max():.4f}")

# СТРАНИЦА 3: Корреляционный анализ
elif page == "Корреляционный анализ":
    st.header("🔗 Корреляционный анализ")
    
    numeric_data = train_data.select_dtypes(include=[np.number])
    
    st.subheader("Тепловая карта корреляции")
    
    fig_size = st.slider("Размер карты", 8, 20, 12)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Корреляционная матрица признаков', fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Топ корреляций с целевой переменной
    if 'Гармония Бессмертия' in correlation_matrix.columns:
        st.subheader("Топ-10 признаков по корреляции с целевой переменной")
        target_corr = correlation_matrix['Гармония Бессмертия'].abs().sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != 'Гармония Бессмертия'][:10]
        
        fig3 = px.bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation='h',
            title='Корреляция признаков с Гармонией Бессмертия',
            labels={'x': 'Абсолютная корреляция', 'y': 'Признак'},
            color=target_corr.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)

# СТРАНИЦА 4: Результаты моделей
elif page == "Результаты моделей":
    st.header("🤖 Результаты моделей машинного обучения")
    
    if metrics_df is not None:
        st.subheader("Сравнение всех моделей")
        st.dataframe(metrics_df.style.highlight_min(subset=['RMSE_test'], color='lightgreen'), 
                    use_container_width=True)
        
        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset = st.selectbox("Фильтр по датасету", 
                                           ['Все'] + list(metrics_df['Dataset'].unique()))
        with col2:
            selected_model = st.selectbox("Фильтр по модели", 
                                         ['Все'] + list(metrics_df['Model'].unique()))
        
        # Применяем фильтры
        filtered_df = metrics_df.copy()
        if selected_dataset != 'Все':
            filtered_df = filtered_df[filtered_df['Dataset'] == selected_dataset]
        if selected_model != 'Все':
            filtered_df = filtered_df[filtered_df['Model'] == selected_model]
        
        # Визуализация метрик
        col1, col2 = st.columns(2)
        
        with col1:
            fig5 = px.bar(
                filtered_df,
                x='Model',
                y='RMSE_test',
                color='Dataset',
                title='RMSE на тестовой выборке',
                barmode='group',
                text_auto='.6f'
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            fig6 = px.bar(
                filtered_df,
                x='Model',
                y='R2_test',
                color='Dataset',
                title='R² на тестовой выборке',
                barmode='group',
                text_auto='.4f'
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        # Лучшая модель
        best_row = metrics_df.loc[metrics_df['RMSE_test'].idxmin()]
        st.success(f"🏆 **Лучшая модель:** {best_row['Model']} на датасете {best_row['Dataset']}")
        st.info(f"**RMSE:** {best_row['RMSE_test']:.6f} | **MAE:** {best_row['MAE_test']:.6f} | **R²:** {best_row['R2_test']:.6f}")
        
        # График сравнения Train vs Test
        st.subheader("Сравнение Train vs Test")
        comparison_data = []
        for _, row in filtered_df.iterrows():
            comparison_data.append({
                'Модель': f"{row['Dataset']} - {row['Model']}",
                'Train': row['RMSE_train'],
                'Test': row['RMSE_test'],
                'Тип': 'Train'
            })
        
        fig7 = px.scatter(
            metrics_df,
            x='RMSE_train',
            y='RMSE_test',
            color='Model',
            symbol='Dataset',
            title='RMSE: Train vs Test (идеальная линия - диагональ)',
            hover_data=['Dataset', 'Model', 'R2_test']
        )
        # Добавляем диагональную линию
        max_val = max(metrics_df['RMSE_train'].max(), metrics_df['RMSE_test'].max())
        fig7.add_shape(
            type='line',
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    else:
        st.warning("⚠️ Метрики не загружены. Используйте демо-данные.")
        demo_metrics = pd.DataFrame({
            'Dataset': ['MinMax', 'Standard', 'No Scale'] * 3,
            'Model': ['Ridge']*3 + ['RandomForest']*3 + ['GradientBoosting']*3,
            'RMSE_test': [0.0067, 0.0057, 0.0055, 0.0018, 0.0013, 0.0014, 0.00074, 0.00082, 0.00091],
            'R2_test': [0.20, 0.43, 0.47, 0.94, 0.97, 0.97, 0.99, 0.99, 0.99]
        })
        st.dataframe(demo_metrics, use_container_width=True)

# СТРАНИЦА 5: Предсказания
elif page == "Предсказания":
    st.header("🔮 Интерактивные предсказания")
    
    if model is None:
        st.error("⚠️ Модель не загружена. Убедитесь, что файл 'best_model.pkl' находится в рабочей директории.")
        st.info("Использую демо-режим с случайными предсказаниями")
    else:
        st.success(f"✅ Модель загружена: {type(model).__name__}")
    
    st.info("Введите параметры портала для предсказания Гармонии Бессмертия")
    
    # Создание формы ввода
    if feature_names and len(feature_names) > 0:
        st.subheader(f"Введите значения для {len(feature_names)} признаков:")
        
        # Динамическое создание полей
        cols_per_row = 3
        num_rows = (len(feature_names) + cols_per_row - 1) // cols_per_row
        
        input_values = {}
        
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(feature_names):
                    feature = feature_names[idx]
                    with cols[j]:
                        # Получаем среднее значение
                        if feature in train_data.columns:
                            default_val = float(train_data[feature].mean())
                            min_val = float(train_data[feature].min())
                            max_val = float(train_data[feature].max())
                        else:
                            default_val = 0.5
                            min_val = 0.0
                            max_val = 1.0
                        
                        input_values[feature] = st.number_input(
                            feature, 
                            value=default_val,
                            min_value=min_val,
                            max_value=max_val,
                            format="%.4f",
                            key=f"input_{idx}"
                        )
    else:
        st.warning("⚠️ Список признаков не загружен")
        input_values = {}
    
    # Кнопка предсказания
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Предсказать Гармонию Бессмертия", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            if model is not None and len(input_values) > 0:
                # Создаем DataFrame
                input_df = pd.DataFrame([input_values])
                
                # Делаем предсказание
                prediction = float(model.predict(input_df)[0])
                
                st.markdown("---")
                st.subheader("📊 Результат предсказания")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Гармония Бессмертия", 'font': {'size': 24}},
                    delta={'reference': 0.95, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.85], 'color': '#ffcccc'},
                            {'range': [0.85, 0.95], 'color': '#ffffcc'},
                            {'range': [0.95, 1], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.95
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Интерпретация
                col1, col2, col3 = st.columns(3)
                with col2:
                    if prediction >= 0.95:
                        st.success(f"### ✅ {prediction:.6f}")
                        st.balloons()
                        st.markdown("**🎉 Портал стабилен!**\n\nОтличное состояние. Все системы функционируют нормально.")
                    elif prediction >= 0.85:
                        st.warning(f"### ⚠️ {prediction:.6f}")
                        st.markdown("**⚙️ Портал требует внимания**\n\nРекомендуется профилактика и мониторинг состояния.")
                    else:
                        st.error(f"### 🚨 {prediction:.6f}")
                        st.markdown("**🆘 КРИТИЧЕСКОЕ СОСТОЯНИЕ!**\n\nТребуется немедленное магическое восстановление портала!")
                
                # Показываем входные данные
                with st.expander("📋 Введенные параметры"):
                    st.json(input_values)
                
            else:
                # Демо-режим
                prediction = 0.975 + np.random.uniform(-0.05, 0.05)
                st.warning(f"⚠️ Демо-предсказание: **{prediction:.6f}**")
                st.info("Модель не загружена, используется случайное значение")
                
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
            st.exception(e)

# Футер
st.markdown("---")
st.markdown("*Дашборд для анализа Портала Бессмертия | Машинное обучение 2025*")

# Sidebar info
if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Информация о модели")
    st.sidebar.write(f"**Тип:** {type(model).__name__}")
    if feature_names:
        st.sidebar.write(f"**Признаков:** {len(feature_names)}")
    if metrics_df is not None:
        best_row = metrics_df.loc[metrics_df['RMSE_test'].idxmin()]
        st.sidebar.write(f"**Датасет:** {best_row['Dataset']}")
        st.sidebar.write(f"**RMSE (test):** {best_row['RMSE_test']:.6f}")
        st.sidebar.write(f"**R² (test):** {best_row['R2_test']:.4f}")
