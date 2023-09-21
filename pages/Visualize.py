import streamlit as st
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

    columnWiEmbed = st.text_input('Nombre de columna con embeddings', 'embeddings')
    columnWiText = st.text_input('Nombre de columna con texto', 'text')

    # dfEmbd["Embedding3d"] = [x[columnWiEmbed][:3] for index, x in dfEmbd.iterrows()]

    if st.button('Generar Gráfico'):
        x = []
        y = []
        z = []
        text = []
        for index, row in dfEmbd.iterrows():
            embX =  row[columnWiEmbed][0]
            embY =  row[columnWiEmbed][1]
            embZ =  row[columnWiEmbed][2]
            txt =  row[columnWiText]
            x.append(embX)
            y.append(embY)
            z.append(embZ)
            text.append(txt)

        dictDfVis = {
            "x": x,
            "y": y,
            "z": z,
            "text": text
        }

        dictDfVis = pd.DataFrame.from_dict(dictDfVis)
        fig = px.scatter_3d(dictDfVis, x='x', y='y', z='z')
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