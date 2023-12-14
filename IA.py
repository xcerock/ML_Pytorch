from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Generando datos
val1 = 0.7
val2 = 1
start = 0
end = 1
step = 0.02

# Generando datos
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = val1 * X + val2

#Separando datos de entrenamiento y validación
train_size = int(0.8 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

val_size = len(X) - train_size

X_val = X[train_size:]
y_val = y[train_size:]

print("Forma de X_train: ", X_train.shape, "\nForma de y_train: ", y_train.shape, "\nForma de X_val: ", X_val.shape, "\nForma de y_val: ", y_val.shape)


# funcion para visualizar los datos

def plot_data(train_data = X_train, train_labels = y_train, val_data = X_val, val_labels = y_val, predictions = None):
    plt.figure(figsize=(10, 7))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("y", fontsize=20)
    plt.xlabel("x", fontsize=20)

    plt.scatter(train_data, train_labels, c='b', s=6, label='Training data')

    plt.scatter(val_data, val_labels, c='g', s=6, label='Validation data')

    if predictions is not None:
        plt.scatter(val_data, predictions, c='r', s=6, label='Predictions')

    plt.legend(prop = {'size': 20})

    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad=True)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias #formula de regresion
    

torch.manual_seed(42)

model = LinearRegressionModel()

print(list(model.parameters()))

# Haciendo predicciones

with torch.inference_mode():
    y_preds = model(X_val)

print(f"Number of testing samples: {len(X_val)}")
print(f"Number of predictions: {len(y_preds)}")
print(f"Predicted values: \n {y_preds}")

# Visualizando las predicciones
plot_data(predictions=y_preds)

# Funcion de perdida
loss_fn = nn.L1Loss()

# Optimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento

torch.manual_seed(42)

epochs = 200

train_loss_values = []
val_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model.train()
    y_preds = model(X_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#testing
    model.eval()
    with torch.inference_mode():
        y_val_preds = model(X_val)
        val_loss = loss_fn(y_val_preds, y_val.type(torch.float))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            val_loss_values.append(val_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {val_loss}")


# Visualizando la perdida
plt.figure(figsize=(10, 7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel("Loss", fontsize=20)
plt.xlabel("Epoch", fontsize=20)
plt.plot(epoch_count, train_loss_values, label='Training loss')
plt.plot(epoch_count, val_loss_values, label='Validation loss')
plt.legend(prop = {'size': 20})
plt.show()

# Ver los parametros del modelo
print("Parametros aprendidos por el modelos (weights, bias):")
print(model.state_dict())
print("\n Valores originales: ")
print(f"weights: {val1}, bias: {val2}")

model.eval()

with torch.inference_mode():
    y_preds = model(X_val)

print("Predicciones:" , y_preds)
plot_data(predictions=y_preds)

# Crear un directorio para guardar el modelo

MODEL_PATH = Path("model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Directorio del modelo y nombre

MODEL_NAME = "first_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Salvando el modelo

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)

# Cargando el modelo

model_loaded = LinearRegressionModel()

model_loaded.load_state_dict(torch.load(f = MODEL_SAVE_PATH))\

model_loaded.state_dict()

# Haciendo predicciones con el modelo cargado

model_loaded.eval()

with torch.inference_mode():
    y_preds = model_loaded(X_val)

print("Predicciones:" , y_preds)

