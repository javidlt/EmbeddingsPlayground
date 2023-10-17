import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Visualizar",
    page_icon="👋",
)

st.write("# Visualizar embeddings")

def verifyEmbeddingsColumn(df, colEmbedName):
    col = np.array(df[colEmbedName].tolist() )
    is_list_column = df[colEmbedName].apply(lambda x: isinstance(x, list)).all()
    print(is_list_column)
    if is_list_column == True:
        return (col.shape[1] == np.array([row.shape[0] for row in col])).all()
    return False


if 'listOfFilesNamesCluster' not in st.session_state:
    st.session_state.listOfFilesNamesCluster = []
if 'listOfDictsCluster' not in st.session_state:
    st.session_state.listOfDictsCluster = []
if 'indexOfDatasetCluster' not in st.session_state:
    st.session_state.indexOfDatasetCluster = 0
if 'uploaded_file_countCluster' not in st.session_state:
    st.session_state.uploaded_file_countCluster = 0
if 'st.session_state.datasetToUseCluster' not in st.session_state:
    st.session_state.datasetToUseCluster = ""

uploaded_fileCount = st.session_state.uploaded_file_countCluster
datasetToUse = st.session_state.datasetToUseCluster

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["json"])
if uploaded_file is not None and (uploaded_file.name not in st.session_state.listOfFilesNamesCluster):
    if st.sidebar.button('usar archivo'):
        uploaded_fileCount = uploaded_fileCount+1

if uploaded_file is not None and (uploaded_fileCount != st.session_state.uploaded_file_countCluster):
    df = pd.read_json(uploaded_file)
    dictEmbd = df.to_dict()
    st.session_state.listOfDictsCluster.append(dictEmbd)
    st.session_state.listOfFilesNamesCluster.append(uploaded_file.name)
    st.session_state.uploaded_file_countCluster = st.session_state.uploaded_file_countCluster+1

if st.session_state.listOfDictsCluster != []:
    st.session_state.datasetToUseCluster = st.sidebar.radio("Dataset a usar", st.session_state.listOfFilesNamesCluster)
    st.session_state.indexOfDatasetCluster = st.session_state.listOfFilesNamesCluster.index(st.session_state.datasetToUseCluster)
    dfEmbd = pd.DataFrame.from_dict(st.session_state.listOfDictsCluster[st.session_state.indexOfDatasetCluster])
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
                columnWitCluster = st.selectbox('Nombre de columna con cluster', column_names)

    # dfEmbd["Embedding3d"] = [x[columnWiEmbed][:3] for index, x in dfEmbd.iterrows()]

    if st.button('Generar Gráfico') and verifyEmbeddingsColumn(dfEmbd, columnWiEmbed):
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