import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(
    page_title="Clusterizador embeddings",
    page_icon="👋",
)

def convert_to_json(df):
    return df.to_json(index=False)

def elbowMethod(columnWiEmbeddings):
    embeddings = np.array(columnWiEmbeddings)
    # print(embeddings)
    # Calcula la inercia para diferentes valores de k (número de clústeres)
    inertia_values = []
    possible_k_values = range(1, 11)  # Prueba k desde 1 hasta 10 clústeres
    progress_text = "Realizando el método"
    my_bar = st.progress(0, text=progress_text)
    tot = len(possible_k_values)
    for index, k in enumerate(possible_k_values):
        pro = int(((index+1)/tot)*100)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
        my_bar.progress(pro, text=progress_text)
    return [possible_k_values, inertia_values]

def clusterKmeans(optimal_k_value,embeddings):
    try: 
        # Aplica K-Means con el número óptimo de clústeres
        kmeans = KMeans(n_clusters=optimal_k_value, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings)

        return cluster_labels
    except:
        return []

st.write("# Generar clusters")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["json"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    dfEmbd = pd.read_json(uploaded_file)
    column_names = list(dfEmbd.columns.values)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            columnWiEmbeddings = st.selectbox('Nombre de columna con embeddings', column_names)
        with col2:
            columnText = st.selectbox('Nombre de columna con texto', column_names)

    colEmbed = dfEmbd[columnWiEmbeddings].tolist()
    step = st.radio("Paso",["**Metodo del codo**", "**Clusterizar**"],)
    if (step == "**Metodo del codo**"):
        if st.button("Generar"):
            resToPlot = elbowMethod(colEmbed)
            # Visualiza el método del codo
            plt.plot(resToPlot[0], resToPlot[1], marker='o')
            plt.xlabel('Número de Clústeres (k)')
            plt.ylabel('Inercia')
            plt.title('Método del Codo para Determinar k')
            st.pyplot(plt)
    else: 
        optimal_k = st.number_input('Número de clusters', min_value=2)
        if st.button('Generar'):
            clusterlabels = clusterKmeans(optimal_k, colEmbed)
            # Agrega las etiquetas de clúster al DataFrame original
            dfEmbd['cluster'] = clusterlabels
            x = [i[0] for i in colEmbed]
            y = [i[1] for i in colEmbed]
            z = [i[2] for i in colEmbed]
            json = convert_to_json(dfEmbd)
            if (columnText == ''):
                txtList = ['No se dio la columna de texto' for i in range(len(colEmbed))]
            else:
                txtList = dfEmbd[columnText].tolist()

            dfToPlot = {
                "X": x,
                "Y": y,
                "Z": z,
                "cluster": dfEmbd["cluster"].tolist(),
                "text": txtList
            }

            dfToPlot = pd.DataFrame.from_dict(dfToPlot)
            # Visualización 3D con Plotly
            fig = px.scatter_3d(dfToPlot, x='X', y='Y', z='Z', color='cluster', hover_data=["text"],
                                 labels={'X': 'Dimensión 1', 'Y': 'Dimensión 2', 'Z': 'Dimensión 3'},
                                 title='Visualización 3D de Embeddings con Colores de Clústeres')
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button(
                "Descargar",
                json,
                "embeddingsClusterizados.json",
                "text/json",
                key='download-json'
            )

else: 
    st.markdown(
    """
    ### Pasos 
    - Subir json con la columna de embeddings
    - Escribir cuál es la columna del embeddings
    - Opcional: Método del codo para ver número de clusters óptimo
    - Seleccionar número de clusters
    - Seleccionar modelo para clusterizar
    - Generar, visualizar y exportar
    """
    )