import torch.nn as nn
import numpy as np
import pandas as pd
import torch

# Cargamos los datos

data = pd.read_csv('Data\DS_Proyecto_01_Datos_Properati.csv')

# Definimos las variables de entrada y salida

x = data[['rooms', 'bedrooms', 'bathrooms', 'surface_total', 'surface_covered', 'price']]
y = data['price']

# Normalizamos los datos

x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

# Convertimos los datos a tensores

x = torch.tensor(x.values, dtype=torch.float32)

y = torch.tensor(y.values, dtype=torch.float32)

# Definimos el modelo



