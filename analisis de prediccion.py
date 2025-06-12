import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

print("=== ANÁLISIS ARIMA DE MERMAS (UNIDADES) - AGRUPADO POR DÍA ===")

# Carga de datos
df = pd.read_excel("C:/Users/mroal/Downloads/mermas_actividad_unidad_2.xlsx", sheet_name="Hoja1")
# Corregir si los números están con coma decimal (ej. "11,1" → 11.1)
df['merma_unidad'] = df['merma_unidad'].astype(str).str.replace(',', '.').astype(float)

# Asegúrate de que los valores de merma sean positivos.
if (df['merma_unidad'] < 0).any():
    df['merma_unidad'] = df['merma_unidad'].abs()
print("Valores de merma ajustados a positivos si eran negativos.")

# Conversión de fechas y limpieza
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df = df.dropna(subset=['fecha'])
df['dia'] = df['fecha'].dt.to_period('D')
print("Días únicos encontrados en el dataset:")
print(df['dia'].value_counts().sort_index())

# Agrupar por día
serie_diaria = df.groupby('dia')['merma_unidad'].sum()

# Convertir el índice a timestamp
serie_diaria.index = serie_diaria.index.to_timestamp()

# --- NUEVO CÓDIGO PARA VISUALIZAR LAS DIFERENCIAS ---
# Calcula la primera diferencia de la serie
serie_diaria_diff = serie_diaria.diff().dropna() # .dropna() elimina el primer NaN

# Visualizar la serie de tiempo diferenciada
plt.figure(figsize=(12, 5))
plt.plot(serie_diaria_diff, marker='o')
plt.title("Mermas por Día (Primera Diferencia)")
plt.xlabel("Día")
plt.ylabel("Diferencia de Merma")
plt.grid(True)
plt.tight_layout()
plt.savefig("mermas_diarias_diferenciadas.png")
plt.show()

print("\n--- ANÁLISIS DE LA SERIE DIFERENCIADA ---")
print("Puedes observar este gráfico para ver si la serie diferenciada se ve más estacionaria (media y varianza constantes).")
print("Los picos en este gráfico representan grandes cambios de un día a otro.")
# ----------------------------------------------------


# Visualizar la serie temporal original (mantenemos esto)
plt.figure(figsize=(12, 5))
plt.plot(serie_diaria, marker='o')
plt.title("Mermas por Día (Unidades)")
plt.xlabel("Día")
plt.ylabel("Merma Unidad")
plt.grid(True)
plt.tight_layout()
plt.savefig("mermas_diarias.png")
plt.show()


# Modelo ARIMA
print("\nEntrenando modelo ARIMA(1,1,1)...")
modelo = ARIMA(serie_diaria, order=(1, 1, 1))
resultado = modelo.fit()

# Mostrar resumen del modelo
print(resultado.summary())

# Pronóstico a 7 días
n_periodos = 7
pred = resultado.get_forecast(steps=n_periodos)
pred_ci = pred.conf_int()

# Visualizar pronóstico
plt.figure(figsize=(12, 5))
plt.plot(serie_diaria, label="Histórico")
plt.plot(pred.predicted_mean, label="Pronóstico", color='green')
plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='green', alpha=0.3)
plt.title("Pronóstico de Mermas - ARIMA(1,1,1) (Diario)")
plt.xlabel("Día")
plt.ylabel("Merma Unidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pronostico_arima_diario.png")
plt.show()

# Interpretación básica
print("\n=== INTERPRETACIÓN BÁSICA ===")
ultima_merma = serie_diaria.iloc[-1]
proxima_merma = pred.predicted_mean.iloc[0]
if ultima_merma != 0:
    variacion = ((proxima_merma - ultima_merma) / abs(ultima_merma)) * 100
    print(f"Última merma registrada: {ultima_merma:.2f}")
    print(f"Pronóstico para el próximo día: {proxima_merma:.2f}")
    print(f"Variación estimada: {variacion:+.2f}%")
else:
    print(f"Última merma registrada: {ultima_merma:.2f}")
    print(f"Pronóstico para el próximo día: {proxima_merma:.2f}")
    print("La última merma fue cero, no se puede calcular la variación porcentual.")


print(df[['fecha', 'merma_unidad']].head(10))
