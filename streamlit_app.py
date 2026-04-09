import re
if not hasattr(re, 'TEMPLATE'):
    re.TEMPLATE = 1
    
import streamlit as st

from utils import carga_modelo, genera

## Página principal

st.title("Generador de imagenes de mariposas")
st.write("Este es un modelo Light GAN")

## Barra lateral
st.sidebar.subheader("Esta mariposa en realidad no existe")
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.caption("Demo creado en vivo")


## Cargando el modelo 
repo_id = "ceyda/butterfly_cropped_uniq1K_512"
modelo_gan = carga_modelo(repo_id)

# Generamos n mariposas
n_mariposas = 4


def corre():
    with st.spinner("Generando"):
        ims = genera(modelo_gan, n_mariposas)
        st.session_state["ims"] = ims

if "ims" not in st.session_state:
    st.session_state["ims"] = None
    corre()

ims = st.session_state["ims"]
corre_botton = st.button(
    "Genera mariposas",
    on_click=corre,
    help="Estamos en vuelo, abrocha tu cinturón"
)

if ims is not None:
    cols = st.columns(n_mariposas)
    for j, im in enumerate(ims):
        i = j % n_mariposas
        cols[i].image(im, use_column_width=True)