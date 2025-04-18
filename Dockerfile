FROM python:3.10

WORKDIR /app

COPY . .

# Fix for scikit-surprise numpy import issue
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip && pip install numpy==1.23.5

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your preprocessing script
RUN python builder_app.py

EXPOSE 5000
RUN cat recommendation.py


CMD ["python","-u", "app.py"]
