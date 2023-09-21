import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import operator

st.set_page_config(
    page_title="Buscar",
    page_icon="👋",
)

st.write("# Buscar en tu archivo")

results = {}
countSearch = 0
embedder = None
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

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            columnWiEmbed = st.text_input('Nombre de columna con embeddings', 'Embedding')
        with col2:
            columnWiText = st.text_input('Nombre de columna con texto', 'full_text')
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
           genre = st.radio("Modelo para embeddings",["**default**", "**Cualquier modelo huggingFace**"],)
        with col2:
            if genre == "**default**":
                option = st.selectbox(
                    'Modelo',
                    ('ggrn/e5-small-v2', 'intfloat/multilingual-e5-small', 'intfloat/e5-small-v2', 'sentence-transformers/all-MiniLM-L6-v2'))
            else: 
                option = st.text_input('Modelo')
    
    queryToSearch = st.text_input('Buscar tweets similares')
    if st.button('Buscar', type="primary"):
                results = search(option, queryToSearch, dfEmbd, columnWiText, columnWiEmbed)

    st.write(results)
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