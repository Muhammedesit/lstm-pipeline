# Python resmi görüntüsünü kullanın
FROM python:3.8

# Uygulama dosyalarını belirtilen dizine kopyalayın
COPY . /my_lstm_project

# Veri dosyasını Docker imajına kopyala
COPY bat_with_temp.csv /my_lstm_project

# Uygulama dizinine çalışma dizini olarak geçin
WORKDIR /my_lstm_project

# Gerekli Python bağımlılıklarını kurun
RUN pip install --no-cache-dir numpy pandas matplotlib scikit-learn tensorflow

# Docker konteyneri çalıştığında çalışacak komut
CMD [ "python", "./run.py" ]
