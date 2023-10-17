import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import operator
import numpy as np

st.set_page_config(
    page_title="Buscar",
    page_icon="👋",
)

st.write("# Buscar en tu archivo")

results = {}
countSearch = 0
embedder = None

def verifyEmbeddingsColumn(df, colEmbedName):
    col = np.array(df[colEmbedName].tolist() )
    is_list_column = df[colEmbedName].apply(lambda x: isinstance(x, list)).all()
    print(is_list_column)
    if is_list_column == True:
        return (col.shape[1] == np.array([row.shape[0] for row in col])).all()
    return False

def dotProduct(embedding1,embedding2):
  result = 0
  for e1, e2 in zip(embedding1, embedding2):
    result += e1*e2
  return result

def search(model, query, df, colText, colEmbedding):
    global countSearch, embedder
    if countSearch == 0:
        embedder = SentenceTransformer(model)
    countSearch+=1
    embeddingInput = embedder.encode(query).tolist()
    listOfTweetsAndSimilarity = []
    for index, row in df.iterrows():
      embeddingRow = row[colEmbedding]
      similarity = dotProduct(embeddingInput, embeddingRow)
      listOfTweetsAndSimilarity.append([row[colText], similarity])
    listOfTweetsAndSimilarity = sorted(listOfTweetsAndSimilarity, key=operator.itemgetter(1), reverse=True)
    listOfTweets = [x[0] for x in listOfTweetsAndSimilarity]
    listOfSim = [x[1] for x in listOfTweetsAndSimilarity]
    dfRet = {
        "texto": listOfTweets,
        "Similitud": listOfSim
    }
    dfRet = pd.DataFrame.from_dict(dfRet)
    return dfRet

if 'listOfFilesNamesSearch' not in st.session_state:
    st.session_state.listOfFilesNamesSearch = []
if 'listOfDictsSearch' not in st.session_state:
    st.session_state.listOfDictsSearch = []
if 'indexOfDatasetSearch' not in st.session_state:
    st.session_state.indexOfDatasetSearch = 0
if 'uploaded_file_countSearch' not in st.session_state:
    st.session_state.uploaded_file_countSearch = 0
if 'resultsToPrint' not in st.session_state:
    st.session_state.resultsToPrint = {}
if 'st.session_state.queryToSearch' not in st.session_state:
    st.session_state.queryToSearch = ""
if 'st.session_state.datasetToUseSearch' not in st.session_state:
    st.session_state.datasetToUseSearch = ""

uploaded_fileCount = st.session_state.uploaded_file_countSearch
datasetToUse = st.session_state.datasetToUseSearch

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["json"])
if uploaded_file is not None and (uploaded_file.name not in st.session_state.listOfFilesNamesSearch):
    if st.sidebar.button('usar archivo'):
        uploaded_fileCount = uploaded_fileCount+1

if uploaded_file is not None and (uploaded_fileCount != st.session_state.uploaded_file_countSearch):
    df = pd.read_json(uploaded_file)
    dictEmbd = df.to_dict()
    st.session_state.listOfDictsSearch.append(dictEmbd)
    st.session_state.listOfFilesNamesSearch.append(uploaded_file.name)
    st.session_state.uploaded_file_countSearch = st.session_state.uploaded_file_countSearch+1

if st.session_state.listOfDictsSearch != []:
    st.session_state.datasetToUseSearch = st.sidebar.radio("Dataset a usar", st.session_state.listOfFilesNamesSearch)
    print(st.session_state.datasetToUseSearch)
    st.session_state.indexOfDatasetSearch = st.session_state.listOfFilesNamesSearch.index(st.session_state.datasetToUseSearch)
    print(st.session_state.indexOfDatasetSearch)
    dfEmbd = pd.DataFrame.from_dict(st.session_state.listOfDictsSearch[st.session_state.indexOfDatasetSearch])
    column_names = list(dfEmbd.columns.values)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.columnWiEmbed = st.selectbox('Nombre de columna con embeddings', column_names)
        with col2:
            st.session_state.columnWiText = st.selectbox('Nombre de columna con texto', column_names)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
           st.session_state.genre = st.radio("Modelo para embeddings",["**default**", "**Cualquier modelo huggingFace**"],)
        with col2:
            if st.session_state.genre == "**default**":
                st.session_state.option = st.selectbox(
                    'Modelo',
                    ('ggrn/e5-small-v2', 'intfloat/multilingual-e5-small', 'intfloat/e5-small-v2', 'sentence-transformers/all-MiniLM-L6-v2'))
            else: 
                st.session_state.option = st.text_input('Modelo')
    
    st.session_state.queryToSearch = st.text_input('Buscar tweets similares')
    if st.button('Buscar', type="primary"):
        if verifyEmbeddingsColumn(dfEmbd, st.session_state.columnWiEmbed):
            results = search(st.session_state.option, st.session_state.queryToSearch, dfEmbd, st.session_state.columnWiText, st.session_state.columnWiEmbed)
            st.session_state.resultsToPrint = results.to_dict()
        else:
            st.markdown("**La columna de embeddings no es valida**")

    dfToPrint = pd.DataFrame.from_dict(st.session_state.resultsToPrint)
    if st.session_state.resultsToPrint != {}:
        if datasetToUse != st.session_state.datasetToUseSearch and datasetToUse!="":
            st.markdown("**Se ha cambiado el dataset con el que estas trabajando, descarga el resultado o se borrará tu avance cuando des click a generar.**")
        st.write(dfToPrint)
else:
    st.markdown(
    """
    ### Pasos 
    - Subir json con embeddings y el texto al que refiere
    - Escribir nombre de columna con embeddings y columna del texto
    - Escribir modelo de para realizar embeddings de hugging face que se uso para realizar los embeddings
    - Buscar en tus datos
    """
    )