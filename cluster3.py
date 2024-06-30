import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import altair as alt

st.title('Análisis de Clustering para Productores de Cereza')

# Subir archivo
uploaded_file = st.file_uploader("Sube tu archivo de Excel", type=["xlsx"])

if uploaded_file:
    # Cargar y mostrar los datos
    data = pd.read_excel(uploaded_file, sheet_name='ANALISIS X GRUPO')
    
    # Cleaning and preparing the data
    data.columns = data.iloc[0]
    data = data.drop(0)
    data = data.dropna(how='all')
    data.reset_index(drop=True, inplace=True)
    
    data.columns = [
        'Razon Social', 'Humidificador', 'Hidrocooler', 'Riego Tecnificado', 'Control Helada', 
        'Buena Gestion Cosecha', 'Mallas Frio', 'Techo Lluvia Macrotunel', 'Sensor Humedad', 'Unused1', 
        'Unused2', 'Unused3', 'Unused4', 'Unused5', 'Unused6', 'Unused7', 'Unused8', 'Unused9', 
        'Unused10', 'Unused11', 'Unused12', '% Entrega', 'Calidad Arboles', 'Calidad Fertilizacion', 
        'Control Malezas', 'Innovacion Variedades', 'Manejo Fitosanitario', 'Calibre Fruta', 
        'Firmeza Fruta', 'Huella Hídrica', 'Analisis Suelo Agua Hojas'
    ]
    
    data = data.drop(columns=['Unused1', 'Unused2', 'Unused3', 'Unused4', 'Unused5', 'Unused6', 'Unused7', 'Unused8', 'Unused9', 'Unused10', 'Unused11', 'Unused12'])
    
    numeric_columns = [
        'Humidificador', 'Hidrocooler', 'Riego Tecnificado', 'Control Helada', 'Buena Gestion Cosecha', 
        'Mallas Frio', 'Techo Lluvia Macrotunel', 'Sensor Humedad', '% Entrega', 'Calidad Arboles', 
        'Calidad Fertilizacion', 'Control Malezas', 'Innovacion Variedades', 'Manejo Fitosanitario', 
        'Calibre Fruta', 'Firmeza Fruta', 'Huella Hídrica', 'Analisis Suelo Agua Hojas'
    ]
    
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data_cleaned = data.dropna()
    columns_to_drop = [
        'Humidificador', 'Hidrocooler', 'Riego Tecnificado', 'Control Helada', 
        'Buena Gestion Cosecha', 'Mallas Frio', 'Techo Lluvia Macrotunel', 'Sensor Humedad'
    ]
    data_cleaned = data.drop(columns=columns_to_drop)
    
    numeric_columns_cleaned = [
        '% Entrega', 'Calidad Arboles', 'Calidad Fertilizacion', 'Control Malezas', 
        'Innovacion Variedades', 'Manejo Fitosanitario', 'Calibre Fruta', 
        'Firmeza Fruta', 'Huella Hídrica', 'Analisis Suelo Agua Hojas'
    ]
    
    # Escalar los datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_cleaned[numeric_columns_cleaned])
    
    # Aplicar K-Means con 3 clusters
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Añadir etiquetas de clusters a los datos originales
    data_cleaned['Cluster'] = clusters
    
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    data_cleaned['PCA1'] = pca_data[:, 0]
    data_cleaned['PCA2'] = pca_data[:, 1]

    # Página de inicio
    if 'page' not in st.session_state:
        st.session_state.page = 'Gráfico de Clusters'

    # Barra lateral para navegación
    st.sidebar.title('Navegación')
    page = st.sidebar.radio("Ir a", ('Gráfico de Clusters', 'Pairplot', 'Gráficos Interactivos', 'Comparación de Clusters'))

    if page == 'Gráfico de Clusters':
        st.session_state.page = 'Gráfico de Clusters'
    elif page == 'Pairplot':
        st.session_state.page = 'Pairplot'
    elif page == 'Gráficos Interactivos':
        st.session_state.page = 'Gráficos Interactivos'
    elif page == 'Comparación de Clusters':
        st.session_state.page = 'Comparación de Clusters'

    # Contenido de la página de gráficos de clusters
    if st.session_state.page == 'Gráfico de Clusters':
        st.subheader('Gráfico de Clusters')
        chart = alt.Chart(data_cleaned).mark_circle(size=60).encode(
            x='PCA1',
            y='PCA2',
            color='Cluster:N',
            tooltip=['Razon Social', 'PCA1', 'PCA2', 'Cluster']
        ).interactive().properties(
            title='Gráfico de Clusters usando PCA'
        )
        st.altair_chart(chart, use_container_width=True)

    # Contenido de la página de pairplot
    elif st.session_state.page == 'Pairplot':
        st.subheader('Pairplot de las Características')
        pairplot_fig = sns.pairplot(data_cleaned[numeric_columns_cleaned + ['Cluster']], hue='Cluster', palette='viridis')
        st.pyplot(pairplot_fig)

    # Contenido de la página de gráficos interactivos
    elif st.session_state.page == 'Gráficos Interactivos':
        st.subheader('Gráficos Interactivos con Altair')
        variable_x = st.selectbox('Seleccione la variable en el eje X', numeric_columns_cleaned)
        variable_y = st.selectbox('Seleccione la variable en el eje Y', numeric_columns_cleaned)
        if variable_x and variable_y and variable_x != variable_y:
            chart = alt.Chart(data_cleaned).mark_circle(size=60).encode(
                x=alt.X(variable_x, scale=alt.Scale(zero=False)),
                y=alt.Y(variable_y, scale=alt.Scale(zero=False)),
                color='Cluster:N',
                tooltip=['Razon Social', variable_x, variable_y, 'Cluster']
            ).interactive().properties(
                title=f'{variable_x} vs {variable_y}'
            )
            st.altair_chart(chart, use_container_width=True)

    # Contenido de la página de comparación de clusters
    elif st.session_state.page == 'Comparación de Clusters':
        st.subheader('Comparación de Clusters')
        cluster_summary = data_cleaned.groupby('Cluster')[numeric_columns_cleaned].mean()
    
        def create_radar_chart(data, title, labels):
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            data = np.concatenate((data,[data[0]]))
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, data, color='blue', alpha=0.25)
            ax.plot(angles, data, color='blue', linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(title, size=15, color='blue', y=1.1)
            for i in range(len(data)-1):
                ax.text(angles[i], data[i] + 0.1, round(data[i], 2), horizontalalignment='center', size=10, color='black', weight='semibold')
            st.pyplot(fig)
    
        cluster_averages = cluster_summary.values
        labels = cluster_summary.columns
    
        for i in range(3):
            create_radar_chart(cluster_averages[i], f'Cluster {i}', labels)
