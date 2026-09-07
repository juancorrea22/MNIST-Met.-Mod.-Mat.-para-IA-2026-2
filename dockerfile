# Usa una imagen oficial ligera de Python
FROM python:3.10-slim

# Directorio de trabajo en el contenedor
WORKDIR /workspace

# Copia e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y scripts al contenedor
COPY . .

# Comando para ejecutar tu script principal (reemplaza main.py por el nombre de tu archivo)
CMD ["python", "main.py"]