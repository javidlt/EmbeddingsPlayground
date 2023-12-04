import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import gudhi

st.set_page_config(
    page_title="Clusterizador embeddings",
    page_icon="👋",
)

def convert_to_json(df):
    return df.to_json(index=False)

def convert_to_csv(df):
    csvToRet = pd.DataFrame.from_dict(df)
    return csvToRet.to_csv(index=False)

def verifyEmbeddingsColumn(df, colEmbedName):
    col = np.array(df[colEmbedName].tolist() )
    is_list_column = df[colEmbedName].apply(lambda x: isinstance(x, list)).all()
    print(is_list_column)
    if is_list_column == True:
        return (col.shape[1] == np.array([row.shape[0] for row in col])).all()
    return False

def elbowMethod(columnWiEmbeddings):
    embeddings = np.array(columnWiEmbeddings)
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

# funciones reducción de dimensionalidades
def genPCA(embeddings, nDims):
    pca = PCA(n_components=nDims)
    X_pca = pca.fit_transform(embeddings)
    return X_pca

def genUMAP(embeddings, nDims):
    umap_model = umap.UMAP(n_components=nDims)
    X_umap = umap_model.fit_transform(embeddings)
    return X_umap

def genTSNE(embeddings, nDims):
    tsne = TSNE(n_components=nDims, random_state=42)  
    X_tsne = tsne.fit_transform(embeddings)
    return X_tsne

def genTDA(embeddings, nDims):
    # Calcula la complejidad de Rips
    rips_complex = gudhi.RipsComplex(points=embeddings, max_edge_length=nDims)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=nDims)

    # Calcula la topología persistente
    persistence = simplex_tree.persistence(min_persistence=0.1)

    # Reducción de dimensionalidad mediante la topología persistente
    reduced_data = np.array([[pt[1], pt[0]] for pt in persistence])
    return reduced_data

def reduceDim(embeddings, nDims, method):
    if method == 'UMAP':
        return genUMAP(embeddings, nDims)
    elif method == 'PCA':
        return genPCA(embeddings, nDims)
    elif method == 'TSNE':
        return genTSNE(embeddings, nDims)
    elif method == 'TDA':
        return genTDA(embeddings, nDims)
    
def clusterAndVisualize(columnDimensionReductionModel,colEmbed,nDimensions, optimal_k,dfEmbd, columnText, nameFile):
    embeddingWithSelectedDimensions = reduceDim(colEmbed, nDimensions, columnDimensionReductionModel)
    clusterlabels = clusterKmeans(optimal_k, embeddingWithSelectedDimensions)
    # Agrega las etiquetas de clúster al DataFrame original
    pro = 6666
    if pro < .99:
        my_bar.progress(pro, text=f"Generando dataset")
    dfEmbd['reducedEmbeddings'] = list(embeddingWithSelectedDimensions)
    dfEmbd['cluster'] = clusterlabels
    json = convert_to_json(dfEmbd)
    csv = convert_to_csv(dfEmbd)
    if (columnText == ''):
        txtList = ['No se dio la columna de texto' for i in range(len(colEmbed))]
    else:
        txtList = dfEmbd[columnText].tolist()
    st.session_state.dfToPlot = {
        "X": embeddingWithSelectedDimensions[:,0],
        "Y": embeddingWithSelectedDimensions[:,1],
        "Z": embeddingWithSelectedDimensions[:,2],
        "cluster": dfEmbd["cluster"].tolist(),
        "text": txtList
    } 
    dfToPlot = pd.DataFrame.from_dict(st.session_state.dfToPlot)
    # Visualización 3D con Plotly
    fig = px.scatter_3d(dfToPlot, x='X', y='Y', z='Z', color='cluster', hover_data=["text"],
                         labels={'X': 'Dimensión 1', 'Y': 'Dimensión 2', 'Z': 'Dimensión 3'},
                         title='Visualización 3D de Embeddings con Colores de Clústeres',
                         color_discrete_sequence=px.colors.sequential.Viridis)
    
    st.plotly_chart(fig, use_container_width=True)
    print(nameFile)

    with st.container ():
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Descargar json",
                json,
                f"{nameFile}_Clusterizado",
                "text/json",
                key='download-json'
            )
        with col2:
            st.download_button(
                "Descargar csv",
                csv,
                f"{nameFile}_Clusterizado",
                "text/csv",
                key='download-csv'
            )

st.write("# Generar clusters")


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
if 'st.session_state.step' not in st.session_state:
    st.session_state.step = ""

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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.columnWiEmbeddings = st.selectbox('Nombre de columna con embeddings', column_names)
        with col2:
            st.session_state.columnText = st.selectbox('Nombre de columna con texto', column_names)
        with col3:
            st.session_state.columnDimensionReductionModel = st.selectbox('Reducción de dimensiones con', ['UMAP', 'PCA', 'TSNE', 'TDA'])

    colEmbed = dfEmbd[st.session_state.columnWiEmbeddings].tolist()
    embeddingWithSelectedDimensions = []
    step = ""
    # if dfEmbd.dtypes[columnWiEmbeddings] == list:
    if verifyEmbeddingsColumn(dfEmbd, st.session_state.columnWiEmbeddings):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.step = st.radio("Paso",["**Metodo del codo**", "**Clusterizar**"],)
            with col2:
                st.session_state.nDimensions = st.number_input('Número de dimensiones a utilizar para clusterizar', min_value=3, max_value=len(colEmbed[0]))


    if (st.session_state.step == "**Metodo del codo**"):
        if st.button("Generar"):
            my_bar = st.progress(0, text=f"Reduciendo dimensiones con {st.session_state.columnDimensionReductionModel}")
            st.session_state.embeddingWithSelectedDimensions = reduceDim(colEmbed, st.session_state.nDimensions, st.session_state.columnDimensionReductionModel)
            my_bar.progress(1, text="Se ha reducido")
            st.session_state.resToPlot = elbowMethod(st.session_state.embeddingWithSelectedDimensions)
            # Visualiza el método del codo
            plt.plot(st.session_state.resToPlot[0], st.session_state.resToPlot[1], marker='o')
            plt.xlabel('Número de Clústeres (k)')
            plt.ylabel('Inercia')
            plt.title('Método del Codo para Determinar k')
            st.pyplot(plt)
    else: 
        st.session_state.optimal_k = st.number_input('Número de clusters', min_value=2)
        if st.button('Generar'):
            clusterAndVisualize(st.session_state.columnDimensionReductionModel,colEmbed,st.session_state.nDimensions, st.session_state.optimal_k,dfEmbd, st.session_state.columnText, st.session_state.listOfFilesNamesCluster[st.session_state.indexOfDatasetCluster])

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