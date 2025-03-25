# Usamos una imagen base de Python 3.11 slim para mantener la imagen ligera
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /cursoIA/Proyecto2

# Instalamos las dependencias del sistema necesarias para compilar algunas librerías
# - build-essential: Incluye herramientas como gcc para compilar dependencias
# - libatlas-base-dev: Necesario para numpy y otras librerías científicas
# - gfortran: Necesario para scipy
# - libopenblas-dev: Proporciona BLAS (Basic Linear Algebra Subprograms) para operaciones numéricas
# - liblapack-dev: Proporciona LAPACK para operaciones de álgebra lineal
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiamos los archivos requirements.txt y los scripts al contenedor
COPY requirements.txt .
COPY train_models.py .
COPY app.py .

# Instalamos las dependencias de Python listadas en requirements.txt
# --no-cache-dir: Evita almacenar caché para mantener la imagen más ligera
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto 8501 para Streamlit
# Streamlit por defecto usa el puerto 8501 para servir la aplicación
EXPOSE 8501

# Definimos el comando por defecto para ejecutar el contenedor
# Este comando se puede sobrescribir al ejecutar el contenedor
CMD ["bash"]