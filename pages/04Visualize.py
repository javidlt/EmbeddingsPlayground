import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import operator
import plotly.express as px

st.set_page_config(
    page_title="Visualizar",
    page_icon="👋",
)

st.write("# Visualizar embeddings")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["json"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # Can be used wherever a "file-like" object is accepted:
    # if uploaded_file.name.endswith('.csv'):
    #     dfEmbd = pd.read_csv(uploaded_file)
    # elif uploaded_file.name.endswith('.xlsx'):
    dfEmbd = pd.read_json(uploaded_file)
    column_names = list(dfEmbd.columns.values)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            columnWiEmbed = st.selectbox('Nombre de columna con embeddings', column_names)
        with col2:
            columnWiText = st.selectbox('Nombre de columna con texto', column_names)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            opt = st.radio("Paso",["**Con clusters**", "**Sin clusters**"],)
        with col2:
            if opt == "**Con clusters**":
                columnWitCluster = st.text_input('Nombre de columna con cluster', 'cluster')

    # dfEmbd["Embedding3d"] = [x[columnWiEmbed][:3] for index, x in dfEmbd.iterrows()]

    if st.button('Generar Gráfico'):
        x = []
        y = []
        z = []
        text = []
        cluster = []
        for index, row in dfEmbd.iterrows():
            embX =  row[columnWiEmbed][0]
            embY =  row[columnWiEmbed][1]
            embZ =  row[columnWiEmbed][2]
            txt =  row[columnWiText]
            clst = 0
            if opt == "**Con clusters**":
                clst = row[columnWitCluster]
            x.append(embX)
            y.append(embY)
            z.append(embZ)
            text.append(txt)
            cluster.append(clst)

        dictDfVis = {
            "x": x,
            "y": y,
            "z": z,
            "text": text,
            "cluster": cluster
        }

        dictDfVis = pd.DataFrame.from_dict(dictDfVis)
        fig = px.scatter_3d(dictDfVis, x='x', y='y', z='z', color='cluster', hover_data=["text"],
                                 labels={'X': 'Dimensión 1', 'Y': 'Dimensión 2', 'Z': 'Dimensión 3'},
                                 title='Visualización 3D de Embeddings con Colores de Clústeres')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(
    """
    ### Pasos 
    - Subir json con embeddings y el texto al que refiere
    - Escribir nombre de columna con embeddings y columna del texto
    """
    )
    # st.write(dataframe)