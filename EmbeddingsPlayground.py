import streamlit as st

st.set_page_config(
    page_title="Embeddings Playground",
    page_icon="👋",
)

st.write("# Prototipo Embeddings 👋")

logo = "https://signalab.mx/wp-content/uploads/2019/07/logo_signa-1.png"

st.markdown(
    """
    ### Toolkit 
    - Generador de embeddings de texto
    - Buscador a partir de embeddings
    - Visualizador de embeddings

    Diseñado para realizar embeddings con cualquier modelo que este en HuggingFace

    [**Leaderboard de modelos para embeddings:**](https://huggingface.co/spaces/mteb/leaderboard)
    """
    )

def add_logo():
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url({logo});
                background-size: 180px;
                width: 300px;
                height: 30px;
                background-repeat: no-repeat;
                padding-top: 70px;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

