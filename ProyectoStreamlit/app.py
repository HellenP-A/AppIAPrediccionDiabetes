import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Cargar el escalador
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar los dos mejores modelos
try:
    model_1 = load_model('Modelo 1_epoch_3.keras')
    model_2 = load_model('Modelo 1_epoch_2.keras')
except FileNotFoundError as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# Barra lateral
st.sidebar.header("📋 Acerca del Proyecto")
st.sidebar.markdown("""
Este proyecto utiliza redes neuronales para predecir el riesgo de diabetes o pre-diabetes en pacientes, basado en el dataset *CDC Diabetes Health Indicators*.  
El objetivo es proporcionar una herramienta interactiva que ayude a identificar riesgos de salud de manera eficiente.
""")

st.sidebar.markdown("### 🛠️ Detalles del Desarrollo")
st.sidebar.markdown("""
- Se entrenaron **cuatro modelos** de redes neuronales, cada uno con una arquitectura y configuración de hiperparámetros única (optimizadores: Adam, SGD, RMSprop; regularización: L2).  
- Se evaluaron métricas de rendimiento (accuracy, precision, recall, F1-score) para cada época de entrenamiento.  
- Se seleccionaron las **2 mejores épocas** de cada modelo según el F1-score, resultando en un total de **8 épocas**.  
- Las 2 épocas con mejor desempeño fueron integradas en esta aplicación para realizar predicciones precisas.
""")

st.sidebar.markdown("### 📚 Lecciones Aprendidas")
st.sidebar.markdown("""
- **Preprocesamiento:** El balanceo de clases con SMOTE es crucial para datasets desbalanceados como este.  
- **Hiperparámetros:** La elección de optimizadores (Adam, SGD, RMSprop) y tasas de aprendizaje impacta significativamente el rendimiento del modelo.  
- **Regularización:** Técnicas como Dropout y L2 fueron efectivas para mitigar el sobreajuste en modelos complejos.  
- **Interfaz:** Streamlit permitió desarrollar esta aplicación web interactiva para visualizar resultados.  
- **Evaluación:** El F1-score resultó ser una métrica clave para problemas de clasificación binaria con clases desbalanceadas.  
""")

st.sidebar.markdown("---")  # Línea divisoria para mejor separación visual
st.sidebar.markdown("Desarrollado para el curso de **Inteligencia Artificial** de la **Universidad CENFOTEC**.")
st.sidebar.markdown("**Profesor:** Rodrigo Herrera Garro")
st.sidebar.markdown("**Autores:** Hellen Aguilar Noguera y Jose Leonardo Araya Parajeles")

# Encabezado personalizado (modificado para evitar redundancia)
st.markdown("""
# Aplicación de Inteligencia Artificial para la Predicción de Diabetes

Desarrollado por:  
**Hellen Aguilar Noguera**  
**Jose Leonardo Araya Parajeles**  
*Universidad CENFOTEC*
""")

# Estilo visual personalizado
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;  /* Fondo gris claro */
}
h1 {
    color: #1e88e5;  /* Título en azul */
}
h2, h3 {
    color: #1565c0;  /* Subtítulos en un azul más oscuro */
}
.stButton>button {
    background-color: #1e88e5;  /* Botón en azul */
    color: white;  /* Texto del botón en blanco */
}
.stButton>button:hover {
    background-color: #1565c0;  /* Color del botón al pasar el mouse */
}
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("🩺 Predicción de Diabetes con Redes Neuronales")
st.write("Esta aplicación utiliza modelos de redes neuronales para predecir si un paciente tiene diabetes o pre-diabetes.")
st.write("Complete los datos del paciente y presione 'Predecir' para obtener el resultado.")

# Información sobre los modelos seleccionados
st.header("Información de los dos mejores Modelos")
st.subheader("Modelos Seleccionados")
st.write(f"Modelo 1: modelo1 (Época 3)")
st.write(f"Modelo 2: modelo1 (Época 2)")

# Formulario para ingresar datos
st.header("Ingrese los Datos del Paciente")

# Definir un diccionario con las categorías de edad
age_categories = {
    1: "18-24 años",
    2: "25-29 años",
    3: "30-34 años",
    4: "35-39 años",
    5: "40-44 años",
    6: "45-49 años",
    7: "50-54 años",
    8: "55-59 años",
    9: "60-64 años",
    10: "65-69 años",
    11: "70-74 años",
    12: "75-79 años",
    13: "80+ años"
}

# Definir un diccionario con las categorías de ingresos
income_categories = {
    1: "Menos de $10,000",
    2: "$10,000 - $14,999",
    3: "$15,000 - $19,999",
    4: "$20,000 - $24,999",
    5: "$25,000 - $34,999",
    6: "$35,000 - $49,999",
    7: "$50,000 - $74,999",
    8: "$75,000 o más"
}

with st.form("patient_form"):
    # Dividir el formulario en tres columnas
    col1, col2, col3 = st.columns(3)

    # Columna 1: Datos de salud
    with col1:
        st.subheader("Datos de Salud")
        HighBP = st.selectbox("¿Tiene presión arterial alta?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        HighChol = st.selectbox("¿Tiene colesterol alto?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        CholCheck = st.selectbox("¿Se ha revisado el colesterol en los últimos 5 años?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        BMI = st.number_input(
            "Índice de Masa Corporal (IMC)",
            min_value=10.0,
            max_value=60.0,
            value=22.0,
            step=0.1,
            help="El IMC se calcula como peso (kg) / altura (m)². Un IMC normal está entre 18.5 y 24.9."
        )
        Smoker = st.selectbox("¿Es fumador?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        Stroke = st.selectbox("¿Ha tenido un accidente cerebrovascular?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        HeartDiseaseorAttack = st.selectbox("¿Ha tenido una enfermedad cardíaca o un ataque al corazón?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

    # Columna 2: Hábitos y estilo de vida
    with col2:
        st.subheader("Hábitos y Estilo de Vida")
        PhysActivity = st.selectbox("¿Realiza actividad física regularmente?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        Fruits = st.selectbox("¿Consume frutas regularmente?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        Veggies = st.selectbox("¿Consume vegetales regularmente?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        HvyAlcoholConsump = st.selectbox("¿Tiene un consumo excesivo de alcohol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        AnyHealthcare = st.selectbox("¿Tiene acceso a atención médica?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        NoDocbcCost = st.selectbox("¿No ha visitado al médico debido a costos?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        GenHlth = st.selectbox("Salud general (1: Excelente, 5: Muy mala)", [1, 2, 3, 4, 5])

    # Columna 3: Datos demográficos
    with col3:
        st.subheader("Datos Demográficos")
        MentHlth = st.number_input("Días con problemas de salud mental (últimos 30 días)", min_value=0, value=0)
        PhysHlth = st.number_input("Días con problemas de salud física (últimos 30 días)", min_value=0, value=0)
        DiffWalk = st.selectbox("¿Tiene dificultad para caminar o subir escaleras?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        Sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        Age = st.selectbox(
            "Edad (seleccione el rango de edad del paciente)",
            options=list(age_categories.keys()),
            format_func=lambda x: age_categories[x],
            help="Seleccione la categoría de edad correspondiente al paciente."
        )
        Education = st.selectbox("Nivel educativo (1: Sin educación formal, 6: Universitario completo)", [1, 2, 3, 4, 5, 6])
        Income = st.selectbox(
            "Ingresos anuales (seleccione el rango en dólares)",
            options=list(income_categories.keys()),
            format_func=lambda x: income_categories[x],
            help="Seleccione el rango de ingresos anuales del paciente en dólares."
        )

    # Botón para realizar la predicción
    submitted = st.form_submit_button("Predecir")

if submitted:
    # Crear un array con los datos ingresados
    input_data = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
        PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
        NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
        Education, Income
    ]])
    
    # Escalar los datos
    input_data_scaled = scaler.transform(input_data)
    
    # Realizar predicciones con ambos modelos
    pred_1 = model_1.predict(input_data_scaled)[0][0]
    pred_2 = model_2.predict(input_data_scaled)[0][0]
    
    # Promediar las predicciones
    avg_pred = (pred_1 + pred_2) / 2
    final_result = "Tiene diabetes o pre-diabetes" if avg_pred > 0.5 else "Está saludable"
    
    # Mostrar el resultado
    st.subheader("Resultado de la Predicción")

    # Convertir la probabilidad promedio a porcentaje
    avg_pred_percentage = avg_pred * 100

    # Determinar la interpretación cualitativa
    if avg_pred <= 0.2:
        probability_interpretation = "baja probabilidad"
    elif avg_pred <= 0.4:
        probability_interpretation = "probabilidad moderada-baja"
    elif avg_pred <= 0.6:
        probability_interpretation = "probabilidad moderada"
    elif avg_pred <= 0.8:
        probability_interpretation = "probabilidad moderada-alta"
    else:
        probability_interpretation = "alta probabilidad"

    # Mostrar el resultado final con interpretación
    if final_result == "Está saludable":
        st.success(f"✅ **Resultado:** {final_result}")
        st.write(f"Según los modelos, el paciente tiene una **{probability_interpretation}** de tener diabetes o pre-diabetes ({avg_pred_percentage:.1f}%).")
    else:
        st.warning(f"⚠️ **Resultado:** {final_result}")
        st.write(f"Según los modelos, el paciente tiene una **{probability_interpretation}** de tener diabetes o pre-diabetes ({avg_pred_percentage:.1f}%). Se recomienda consultar a un médico.")
    st.write(f"Probabilidad del Modelo 1 (Época 3): {pred_1 * 100:.1f}%")
    st.write(f"Probabilidad del Modelo 2 (Época 2): {pred_2 * 100:.1f}%")
    st.write(f"**Probabilidad promedio:** {avg_pred_percentage:.1f}% (umbral de decisión: 50%)")

# Pie de página
st.markdown("""
---
**© 2025 Hellen Aguilar Noguera y Jose Leonardo Araya Parajeles**  
Desarrollado para el curso de Inteligencia Artificial, Universidad CENFOTEC.
""")