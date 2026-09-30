FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Upgrade pip once
RUN pip install --upgrade pip

# Install core data science libraries
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.1

# Install ML & processing libraries
RUN pip install --no-cache-dir scikit-learn==1.4.2 scipy

# Install remaining dependencies from requirements.txt (Streamlit, Excel tools, etc.)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]