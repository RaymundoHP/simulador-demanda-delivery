# ==========================================================
# Proyecto: Análisis de Restaurantes Zomato
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Cargar datasets
# ===============================
zomato = pd.read_csv("archive (7)/zomato.csv", encoding='latin-1')
country = pd.read_excel("archive (7)/Country-Code.xlsx")

print("Archivos cargados correctamente ✅")

# ===============================
# 2. Unir datasets
# ===============================
zomato = zomato.merge(country, on='Country Code', how='left')

# ===============================
# 3. Limpieza básica
# ===============================
zomato.drop_duplicates(inplace=True)
zomato.dropna(subset=['Cuisines', 'Aggregate rating'], inplace=True)

# ===============================
# 4. Análisis exploratorio
# ===============================

# Top 10 países con más restaurantes
top_countries = zomato['Country'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='crest')
plt.title("Top 10 países con más restaurantes en Zomato")
plt.xlabel("Cantidad de restaurantes")
plt.ylabel("País")
plt.tight_layout()
plt.show()

# Tipos de cocina más populares
top_cuisines = zomato['Cuisines'].value_counts().head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title("Tipos de cocina más populares")
plt.xlabel("Cantidad")
plt.ylabel("Cocina")
plt.tight_layout()
plt.show()

# Relación entre precio y calificación
plt.figure(figsize=(7,4))
sns.scatterplot(data=zomato, x='Average Cost for two', y='Aggregate rating', alpha=0.4)
plt.title("Relación entre costo promedio y calificación")
plt.xlabel("Costo promedio para dos")
plt.ylabel("Calificación promedio")
plt.tight_layout()
plt.show()

# ===============================
# 5. Exportar dataset limpio
# ===============================
zomato.to_csv("zomato_clean.csv", index=False)
print("✅ Dataset limpio exportado: zomato_clean.csv")
