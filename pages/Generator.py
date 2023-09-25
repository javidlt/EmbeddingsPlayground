import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import operator

st.set_page_config(
    page_title="Generador",
    page_icon="👋",
)

st.write("# Generar embeddings")

def generate(mod, df, colText):
    embedder = SentenceTransformer(mod)
    tot = len(df)
    dfW = df
    dfW["Embedding"] = None
    progress_text = "Generando embeddings"
    my_bar = st.progress(0, text=progress_text)
    for index, row in dfW.iterrows():
        pro = int(((index+1)/tot)*100)
        embedding = embedder.encode(str(row[colText])).tolist()
        dfW.at[index, 'Embedding'] = embedding
        my_bar.progress(pro, text=progress_text)


    return dfW

def convert_to_json(df):
    return df.to_json(index=False)


uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "excel", "json"])
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    if uploaded_file.name.endswith('.csv'):
        dfEmbd = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        dfEmbd = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        dfEmbd = pd.read_json(uploaded_file)


    # columnWiText = st.text_input('Nombre de columna con texto', 'text')
    column_names = list(dfEmbd.columns.values)
    columnWiText = st.selectbox('Nombre de columna con texto', column_names)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
           type = st.radio("Modelo para embeddings",["**default**", "**Cualquier modelo huggingFace**"],)
        with col2:
            if type == "**default**":
                model = st.selectbox(
                    'Modelo',
                    ('ggrn/e5-small-v2', 'intfloat/multilingual-e5-small', 'intfloat/e5-small-v2', 'sentence-transformers/all-MiniLM-L6-v2'))
            else: 
                model = st.text_input('Modelo')
    if st.button('Generar embeddings', type="primary"):
        dfFinal = generate(model, dfEmbd,columnWiText)
        json = convert_to_json(dfFinal)
        st.download_button(
            "Descargar",
            json,
            "embeddingsTexto.json",
            "text/json",
            key='download-json'
        )
        st.write(dfFinal)
else:
    st.markdown(
    """
    ### Pasos 
    - Subir json, csv o excel con la columna de texto con la que deseas generar los embeddings
    - Escribir cuál es la columna del texto
    - Seleccionar el modelo con el que se harán los embeddings
    - Exportar json con tus embeddings para usarlo
    """
    )