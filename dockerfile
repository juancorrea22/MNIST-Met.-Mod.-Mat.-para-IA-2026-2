# Usa una imagen oficial ligera de Python
FROM python:3.10-slim

# Directorio de trabajo en el contenedor
WORKDIR /workspace

# Copia e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y scripts al contenedor
COPY . .

# Puerto por defecto de Streamlit
EXPOSE 8501

# Corre la interfaz web. --server.address=0.0.0.0 es necesario para que
# sea accesible desde fuera del contenedor (por ejemplo, desde el celular
# en la misma red durante la demo).
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]