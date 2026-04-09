import re
if not hasattr(re, 'TEMPLATE'):
    re.TEMPLATE = 1
    
import streamlit as st

from utils import carga_modelo, genera

## Página principal

st.title("Generador de imagenes")
st.write("Este es un modelo Light GAN")
st.write("Las Redes Generativas Adversariales (GANs) son un tipo de modelo de deep learning compuesto por dos redes neuronales que compiten entre sí: un generador, que crea imágenes sintéticas, y un discriminador, que intenta distinguirlas de las reales. A través de este entrenamiento adversarial, el generador aprende a producir imágenes cada vez más realistas. Este demo utiliza el modelo Lightweight GAN ceyda/butterfly_cropped_uniq1K_512, entrenado por Ceyda Cinarel sobre un conjunto de datos curado de aproximadamente 1,000 imágenes únicas de mariposas del Smithsonian, generando imágenes de 512×512 píxeles.")
st.write("Haz clic en Generar para crear una mariposa que no existe en la naturaleza")

## Barra lateral
st.sidebar.subheader("Esta mariposa en realidad no existe")
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.caption("Demo creado en vivo")


## Cargando el modelo 
repo_id = "ceyda/butterfly_cropped_uniq1K_512"
modelo_gan = carga_modelo(repo_id)

# Generamos n mariposas
n_mariposas = 12


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
    help="Estamos generando las imágenes, abrocha tu cinturón"
)

if ims is not None:
    cols = st.columns(n_mariposas)
    for j, im in enumerate(ims):
        i = j % n_mariposas
        cols[i].image(im, use_container_width=True)