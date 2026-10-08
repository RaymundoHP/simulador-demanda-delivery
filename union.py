# ==========================================================
# Proyecto: Predicción de Demanda de Ventas y Delivery de Alimentos
# Dataset: Food Demand Dataset (versión 2)
# ==========================================================

import pandas as pd
import numpy as np

# ===============================
# 1. Cargar los archivos
# ===============================
train_path = "archive (5)/foodDemand_train/train.csv"
center_path = "archive (5)/foodDemand_train/fulfilment_center_info.csv"
meal_path = "archive (5)/foodDemand_train/meal_info.csv"
test_path = "archive (5)/food_Demand_test.csv"
sample_path = "archive (5)/sample_submission.csv"

print("Cargando datasets...")
train = pd.read_csv(train_path)
centers = pd.read_csv(center_path)
meals = pd.read_csv(meal_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("Archivos cargados correctamente ✅")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ===============================
# 2. Unir la información
# ===============================
print("\nUniendo datasets...")
df = train.merge(centers, on="center_id", how="left")
df = df.merge(meals, on="meal_id", how="left")

print(f"Dataset final: {df.shape}")

# ===============================
# 3. Limpieza básica
# ===============================
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# ===============================
# 4. Feature Engineering
# ===============================
# Tasa de descuento
df["discount_rate"] = 1 - (df["checkout_price"] / df["base_price"])

# Promociones activas
df["promo_active"] = ((df["emailer_for_promotion"] + df["homepage_featured"]) > 0).astype(int)

# Precio logarítmico
df["log_price"] = np.log1p(df["checkout_price"])

# ===============================
# 5. Validación de datos
# ===============================
print("\nColumnas del dataset combinado:")
print(df.columns.tolist())

print("\nResumen estadístico:")
print(df.describe())

# ===============================
# 6. Exportar dataset limpio
# ===============================
df.to_csv("food_demand_v2_clean.csv", index=False)
print("\n✅ Archivo exportado correctamente: food_demand_v2_clean.csv")
