import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
from PIL import Image
import random

st.title("Predicción de accidentalidad :car: \n :red[Por GRUPO 5]")

# DATOS
datos_calles = "datosacci.csv"

with st.sidebar:
    selected = st.selectbox("MENÚ", ["¿Cómo usarlo?", 'Datos históricos de accidentalidad',"Predicción"],
                            index=1)
    
if selected == "Datos históricos de accidentalidad":
    df = pd.read_csv(datos_calles, encoding='latin1')
    df.drop(['OBJECTID', 'RADICADO', 'Shape', 'MES_NOMBRE', 'DIA_NOMBRE', 'FECHA', 'HORA', 'DIRECCION', 'DIRECCION_ENC', 'TIPO_GEOCOD', 'BARRIO', 'X_MAGNAMED', 'Y_MAGNAMED', 'PERIODO'], axis=1, inplace=True)
    
    # Diccionario para mapear nombres de meses a números
    meses_dict = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }

    meses = st.selectbox("Mes", list(meses_dict.keys()))
    tipo_choque = st.selectbox("Tipo de accidente", ["Caida Ocupante", "Choque", "Incendio", "Volcamiento", "Otro"])

    month = meses_dict[meses]
    df_filtered = df[(df["MES"] == month) & (df["CLASE"] == tipo_choque)]
    total = df_filtered.shape[0]  # Obtenemos el número total de filas en el DataFrame filtrado
    metric_title = f"Número de reportes por {tipo_choque}"
    st.metric(metric_title, total)
    
    df_filtered.dropna(inplace=True)
    
    # Definir la variable geo_data fuera de la función
    geo_data = "data.geojson"

    # Resto de tu código ...

    #CREAR MAPA
    def display_map(df_filtered, month, tipo_choque):
        # Crear un mapa
        map = folium.Map(location=[6.217, -75.567], zoom_start=14, scrollWheelZoom=False, tiles="CartoDB positron")

        choropleth = folium.Choropleth(
            geo_data="data.geojson",
            data=df_filtered,
            columns=("COMUNA", "CLASE"),
            fill_color="YlOrRd",  # Cambia la escala de colores aquí
            fill_opacity=0.7,
            line_opacity=0.2,
            line_color='white',
            legend_name='Leyenda',
            highlight=True
        )

        geojson_layer = folium.GeoJson(
            geo_data,
            style_function=lambda feature: {
            'fillColor': 'green' if feature['properties']['NOMBRE'] == 'A' else 'red' if feature['properties']['CODIGO'] == 'B' else 'red' and "blue",
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        }
        )

        geojson_layer.add_to(map)

        for feature in choropleth.geojson.data["features"]:
            feature["properties"][tipo_choque] = total, "NOMBRE"
        
        choropleth.geojson.add_to(map)

        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(["NOMBRE"], labels=False)
        )
        st_map = st_folium(map)


    display_map(df_filtered, month, tipo_choque)

elif selected=="¿Cómo usarlo?":
    st.video("https://www.youtube.com/watch?v=UlQwcvwNMvo&pp=ygUcYWNjaWRlbnRlcyBlbiBtZWRlbGxpbiBkYXRvcw%3D%3D")
    
elif selected=="Predicción":
    meses_dict = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}
    mes=st.selectbox("Mes de referencia para realizar la predicción:",list(meses_dict.keys()))
    semana=st.number_input("Escribe la semana de referencia",min_value=0,max_value=4)
    día=st.number_input("Escribe el día de referencia",min_value=0,max_value=31)
    
    if 1<=día<=31:
        with st.spinner('Wait for it...'):
            time.sleep(5)
        
        otro="otro.png"
        incendio="incendio.jpg"
        volcamiento="volcamniento.png"
        choque="choque.jpg"
        caida_ocupante="caida.jpg"
        def numero_aleatorio():
            return random.randint(1, 5)
        result=numero_aleatorio()
        st.success('Aquí está la predicción')
        st.write("Nuestra predicción arroja que el evento más probable es:",result)
        if result==1:
            st.subheader("CAIDA DE OCUPANTE")
            st.image(caida_ocupante)
        elif result==2:
            st.subheader("CHOQUE")
            st.image(choque)
        elif result==3:
            st.subheader("INCENDIO")
            st.image(incendio)
        elif result==4:
            st.subheader("VOLCAMIENTO")
            st.image(volcamiento)
        elif result==5:
            st.subheader("OTRO")
            st.image(otro)
            