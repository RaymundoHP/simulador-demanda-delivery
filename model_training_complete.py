#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_training_complete_optionC.py
Opción C (final, optimizada): integración de TODOS los datasets con precomputación,
reducción de memoria, PCA por bloques (IncrementalPCA), clustering y entrenamiento.
Autor: generado por ChatGPT (adaptado para Ray)
Ejecutar con el virtualenv activado.
"""

import os
import glob
import math
import joblib
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional models (will be used if installed)
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

warnings.filterwarnings("ignore")
np.random.seed(42)

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
PROCESSED_DIR = "processed"
MODELS_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_NAME = "num_orders"                 # objetivo esperado en dataset principal
INDEX_KEYS_CANDIDATES = ["center_id", "meal_id", "week", "date", "city", "center_code"]
FAST_DEBUG = False                         # True -> reduce datos y modelos para debug rápido
CHUNK_ROWS = 200_000                       # tamaño chunk para procesamiento en disco
IPCA_BATCH = 5000                          # batch_size para IncrementalPCA

# If debug, reduce heavy parameters
if FAST_DEBUG:
    RF_N_EST = 50
    XGB_N_EST = 50
    LGB_N_EST = 50
    SAMPLE_FRACTION = 0.05
else:
    RF_N_EST = 150
    XGB_N_EST = 150
    LGB_N_EST = 150
    SAMPLE_FRACTION = 1.0

# -------------------------
# Helper functions
# -------------------------
def list_csv_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"📥 Buscando CSVs en '{data_dir}' -> {len(files)} archivos encontrados")
    return files

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory by converting dtypes."""
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type):
            # Convert short object columns to category
            if df[col].nunique(dropna=False) < 0.5 * len(df):
                try:
                    df[col] = df[col].astype('category')
                except Exception:
                    pass
    return df

def safe_read_csv(path: str, low_memory=True) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=low_memory)
        df = downcast_df(df)
        print(f"   ✅ Cargado: {os.path.basename(path)} -> {df.shape}")
        return df
    except Exception as e:
        print(f"   ❌ Error leyendo {path}: {e}")
        return pd.DataFrame()

def agg_numeric_stats(df: pd.DataFrame, groupby: List[str], numeric_cols: List[str], prefix: str) -> pd.DataFrame:
    """Aggregate numeric columns by groupby columns using common stats."""
    if not numeric_cols or not groupby:
        return pd.DataFrame()
    agg_dict = {}
    for c in numeric_cols:
        agg_dict[c] = ['mean', 'std', 'min', 'max']
    g = df.groupby(groupby).agg(agg_dict)
    # flatten columns
    g.columns = [f"{prefix}_{col}_{stat}" for col, stat in g.columns]
    g = g.reset_index()
    g = downcast_df(g)
    return g

def add_global_stats_as_constants(base: pd.DataFrame, aux: pd.DataFrame, numeric_cols: List[str], prefix: str) -> pd.DataFrame:
    """Compute global stats (mean/std/min/max) and add as new columns to base (same value for all rows)."""
    if aux is None or len(aux)==0 or not numeric_cols:
        return base
    stats = {}
    for c in numeric_cols:
        if c in aux.columns:
            stats[f"{prefix}_{c}_mean"] = float(aux[c].mean())
            stats[f"{prefix}_{c}_std"] = float(aux[c].std())
            stats[f"{prefix}_{c}_min"] = float(aux[c].min())
            stats[f"{prefix}_{c}_max"] = float(aux[c].max())
    # attach constants
    for k, v in stats.items():
        base[k] = v
    return base

def get_common_keys(df_cols: set) -> List[str]:
    keys = [k for k in INDEX_KEYS_CANDIDATES if k in df_cols]
    return keys

# -------------------------
# 1) Load all CSVs
# -------------------------
csv_files = list_csv_files(DATA_DIR)
if len(csv_files) == 0:
    raise SystemExit("No hay archivos CSV en la carpeta 'data/'. Añade tus datasets y vuelve a ejecutar.")

datasets = {}
for p in csv_files:
    name = os.path.basename(p)
    datasets[name] = safe_read_csv(p)

# -------------------------
# 2) Identify base dataset (the one that contains TARGET_NAME)
# -------------------------
base_name = None
for name, df in datasets.items():
    if TARGET_NAME in df.columns:
        base_name = name
        break

if base_name is None:
    # fallback: ask user (here we pick largest numeric-containing file)
    cand = max(datasets.items(), key=lambda kv: kv[1].shape[0] if isinstance(kv[1], pd.DataFrame) else 0)
    base_name = cand[0]
    print(f"⚠ No se detectó `{TARGET_NAME}`. Se usará '{base_name}' como base (fallback).")

base_df = datasets[base_name].copy()
print(f"\n🏁 Dataset base: {base_name} -> {base_df.shape}")

# -------------------------
# 3) Preprocess base: basic cleaning + ensure id keys present
# -------------------------
# Normalize column names
base_df.columns = [c.strip() for c in base_df.columns]

# ensure index keys exist or create an id
common_keys = get_common_keys(set(base_df.columns))
if len(common_keys) == 0:
    print("⚠ No se detectaron keys estándares en base. Se creará columna id_pred")
    base_df = base_df.reset_index().rename(columns={"index": "id_pred"})
    common_keys = ["id_pred"]

print("🔎 Índice de predicción detectado (base):", common_keys)

# standardize date -> week if available
if "date" in base_df.columns and "week" not in base_df.columns:
    try:
        base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
        base_df["week"] = base_df["date"].dt.isocalendar().week
        print("   ➤ Derivada 'week' desde 'date'")
    except Exception:
        pass

# fill target numeric
if TARGET_NAME in base_df.columns:
    base_df[TARGET_NAME] = pd.to_numeric(base_df[TARGET_NAME], errors="coerce").fillna(0)
else:
    # If fallback pick, choose a numeric col as target (last numeric)
    num_cols = [c for c in base_df.columns if pd.api.types.is_numeric_dtype(base_df[c])]
    if not num_cols:
        raise SystemExit("No numeric columns found in base dataset to be used as target.")
    TARGET_NAME = num_cols[-1]
    base_df[TARGET_NAME] = pd.to_numeric(base_df[TARGET_NAME], errors="coerce").fillna(0)
    print(f"⚠ Usando '{TARGET_NAME}' como target alternativo")

# Save initial snapshot
base_snapshot_path = os.path.join(PROCESSED_DIR, f"base_snapshot_{base_name}")
base_df.to_csv(base_snapshot_path + ".csv", index=False)
joblib.dump(base_df.head(1000), base_snapshot_path + ".pkl")  # small sample
print(f"   ✅ Base snapshot saved: {base_snapshot_path}.csv")

# -------------------------
# 4) Feature enrichment: for each auxiliary dataset decide safe merge or global features
# -------------------------
print("\n🔧 Enriqueciendo base con datasets auxiliares (sin duplicar filas)...")

# Define which dataset names we treat as 'by-week' merges (special-case)
# If dataset includes at least one of base common keys, we'll compute groupby stats on that key and merge.
aux_dfs = {k: v for k,v in datasets.items() if k != base_name}

# We'll collect features to add to base_df as new columns
enriched = base_df.copy()

for name, aux in aux_dfs.items():
    if aux is None or aux.empty:
        print(f"   ⚠ {name} vacío, omitiendo")
        continue

    print(f"\n   📥 Procesando: {name} -> {aux.shape}")

    aux_cols = set(aux.columns)
    keys = get_common_keys(aux_cols)

    # numeric columns in aux
    numeric_cols = [c for c in aux.columns if pd.api.types.is_numeric_dtype(aux[c])]
    # limit numeric cols to reasonable number
    if len(numeric_cols) > 40:
        numeric_cols = numeric_cols[:40]

    # Case A: aux contains a key in common with base -> aggregate by that key and merge
    merge_keys = [k for k in common_keys if k in aux_cols]
    if merge_keys:
        # choose best key subset; prefer (center_id, meal_id, week) intersection
        chosen_key = None
        for try_keys in (["center_id","meal_id","week"], ["center_id","week"], ["center_id"], ["city","week"]):
            if all(k in aux_cols for k in try_keys):
                chosen_key = try_keys
                break
        if chosen_key is None:
            chosen_key = [merge_keys[0]]  # fallback to first common key

        chosen_key = chosen_key if isinstance(chosen_key, list) else [chosen_key]
        print(f"      ➤ Agregando por key(s): {chosen_key}")

        # compute aggregations on aux
        try:
            agg = agg_numeric_stats(aux, chosen_key, numeric_cols, prefix=os.path.splitext(name)[0].replace(" ", "_"))
            if not agg.empty:
                # Merge with enriched on chosen_key(s)
                enriched = enriched.merge(agg, on=chosen_key, how="left")
                print(f"      ✅ Merge realizado: agregadas {len(agg.columns)-len(chosen_key)} columnas")
            else:
                print("      ℹ️ No se generaron agregaciones (agg vacío).")
        except Exception as e:
            print(f"      ❌ Error agregando {name}: {e}")
        continue

    # Case B: aux lacks keys -> compute global stats and add as constants or low-cardinality features
    try:
        print(f"      ➤ No tiene keys comunes con base. Calculando stats globales como features constantes.")
        enriched = add_global_stats_as_constants(enriched, aux, numeric_cols, prefix=os.path.splitext(name)[0].replace(" ", "_"))
        print(f"      ✅ Agregadas features globales desde {name}")
    except Exception as e:
        print(f"      ❌ Error generando features globales para {name}: {e}")

# Final downcast
enriched = downcast_df(enriched)
enriched_path = os.path.join(PROCESSED_DIR, "metadataset_enriched.csv")
enriched.to_csv(enriched_path, index=False)
print(f"\n   ✅ Metadataset enriquecido guardado en: {enriched_path}")
print("   shape enriched:", enriched.shape)

# -------------------------
# 5) Feature selection & safe cleaning
# -------------------------
print("\n✨ Limpieza final y selección de features")

# remove high-cardinality object columns (keep categories or convert via target encoding as fallback)
obj_cols = [c for c in enriched.columns if enriched[c].dtype == 'object']
# keep only short object cols
for c in obj_cols:
    nun = enriched[c].nunique(dropna=False)
    if nun > 200:  # too many unique -> drop
        enriched.drop(columns=[c], inplace=True)
    else:
        # convert to category for memory saving
        try:
            enriched[c] = enriched[c].astype("category")
        except Exception:
            pass

# separate features and target
all_features = [c for c in enriched.columns if c != TARGET_NAME]
X_all = enriched[all_features].copy()
y_all = enriched[TARGET_NAME].copy()

# fill na numeric with 0, categorical with mode
for c in X_all.columns:
    if pd.api.types.is_numeric_dtype(X_all[c]):
        X_all[c] = X_all[c].fillna(0)
    else:
        X_all[c] = X_all[c].fillna(X_all[c].mode().iloc[0] if not X_all[c].mode().empty else 0)

# final downcast
X_all = downcast_df(X_all)

# -------------------------
# 6) Train/test split (must be before any global transforms like PCA)
# -------------------------
print("\n📚 Split train/test (SIN transformaciones posteriores que cambien filas)")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42
)

print("   Train:", X_train_raw.shape, "Test:", X_test_raw.shape)
# Quick consistency assert
assert X_train_raw.shape[0] == y_train.shape[0]
assert X_test_raw.shape[0] == y_test.shape[0]

# optional sampling for FAST_DEBUG
if FAST_DEBUG and SAMPLE_FRACTION < 1.0:
    X_train_raw = X_train_raw.sample(frac=SAMPLE_FRACTION, random_state=42)
    y_train = y_train.loc[X_train_raw.index]
    X_test_raw = X_test_raw.sample(frac=SAMPLE_FRACTION, random_state=42)
    y_test = y_test.loc[X_test_raw.index]
    print("   ⚡ FAST_DEBUG active -> sampled training data")

# -------------------------
# 7) Prepare numeric block groups for IPCA
# -------------------------
print("\n🧠 Preparando bloques para PCA por segmentos")

# identify numeric feature groups by simple heuristics: prefixes from aggregated datasets
numeric_cols = [c for c in X_train_raw.columns if pd.api.types.is_numeric_dtype(X_train_raw[c])]
# group by prefix before first underscore (e.g., balaji_week_order_mean -> balaji)
from collections import defaultdict
groups = defaultdict(list)
for c in numeric_cols:
    prefix = c.split("_")[0] if "_" in c else "main"
    groups[prefix].append(c)

# Keep group order deterministic
groups = dict(sorted(groups.items(), key=lambda kv: (0 if kv[0]=='main' else 1, kv[0])))

print("   Encontrados grupos numéricos (por prefijo):", {k: len(v) for k, v in groups.items()})

# For each group, we'll run IncrementalPCA (if group size > 3) else keep raw
pca_components_by_group = {}
X_train_blocks = {}
X_test_blocks = {}

for group_name, cols in groups.items():
    Xg_train = X_train_raw[cols].astype('float32').values
    Xg_test = X_test_raw[cols].astype('float32').values
    # If few features, keep them
    if Xg_train.shape[1] <= 3:
        # just keep raw numeric columns
        X_train_blocks[group_name] = pd.DataFrame(Xg_train, columns=[f"{group_name}__{c}" for c in cols], index=X_train_raw.index)
        X_test_blocks[group_name] = pd.DataFrame(Xg_test, columns=[f"{group_name}__{c}" for c in cols], index=X_test_raw.index)
        continue

    # Standardize group
    scaler_g = StandardScaler()
    Xg_train_s = scaler_g.fit_transform(Xg_train)
    Xg_test_s = scaler_g.transform(Xg_test)

    # Decide n_components: min(8, n_features) but not exceed 0.9 explained variance budget surrogate
    n_feats = Xg_train_s.shape[1]
    n_comp = min(8, n_feats)

    ipca = IncrementalPCA(n_components=n_comp, batch_size=IPCA_BATCH)
    # Fit incrementally - in memory it's fine; if too large, we could stream by chunks
    ipca.fit(Xg_train_s)
    Xg_tr_comp = ipca.transform(Xg_train_s)
    Xg_te_comp = ipca.transform(Xg_test_s)

    # Add components as columns
    comp_cols = [f"{group_name}_PC{i+1}" for i in range(Xg_tr_comp.shape[1])]
    X_train_blocks[group_name] = pd.DataFrame(Xg_tr_comp, columns=comp_cols, index=X_train_raw.index)
    X_test_blocks[group_name] = pd.DataFrame(Xg_te_comp, columns=comp_cols, index=X_test_raw.index)

    # store ipca and scaler
    pca_components_by_group[group_name] = {"ipca": ipca, "scaler": scaler_g, "cols": cols}
    print(f"   ➤ Grupo {group_name}: {n_feats} feats -> IPCA {n_comp} comp")

# -------------------------
# 8) Compose final feature matrices
# -------------------------
print("\n🔗 Componiendo matriz final de features (concatenando bloques y pequeñas columnas)")

# Also include any non-numeric but converted columns (categories encoded to codes)
# Convert categorical columns in X_train_raw to codes (safe)
cat_cols_train = [c for c in X_train_raw.columns if not pd.api.types.is_numeric_dtype(X_train_raw[c])]
for c in cat_cols_train:
    try:
        X_train_raw[c] = X_train_raw[c].astype('category').cat.codes
        X_test_raw[c] = X_test_raw[c].astype('category').cat.codes
    except Exception:
        X_train_raw[c] = X_train_raw[c].fillna(0).astype('int32')
        X_test_raw[c] = X_test_raw[c].fillna(0).astype('int32')

# Now collect all blocks
blocks_train = []
blocks_test = []
for k in X_train_blocks.keys():
    blocks_train.append(X_train_blocks[k])
    blocks_test.append(X_test_blocks[k])

# Also add remaining numeric columns that were not consumed in groups? groups covered numeric_cols so they are included.
# For safety, include any columns not present in blocks by name prefix
trained_cols = set().union(*[set(df.columns) for df in blocks_train]) if blocks_train else set()
leftover_cols = [c for c in X_train_raw.columns if c not in trained_cols]
if leftover_cols:
    blocks_train.append(X_train_raw[leftover_cols])
    blocks_test.append(X_test_raw[leftover_cols])

X_train_final = pd.concat(blocks_train, axis=1).fillna(0)
X_test_final = pd.concat(blocks_test, axis=1).fillna(0)

print("   Final shapes:", X_train_final.shape, X_test_final.shape)
# ensure numeric dtype
for c in X_train_final.columns:
    X_train_final[c] = pd.to_numeric(X_train_final[c], errors='coerce').fillna(0)
for c in X_test_final.columns:
    X_test_final[c] = pd.to_numeric(X_test_final[c], errors='coerce').fillna(0)

# -------------------------
# 9) Optional clustering features (fit on train PCA concatenation)
# -------------------------
print("\n🔍 Agregando features de clustering (KMeans)")

# Use a small dense representation for clustering: take first 6 columns of X_train_final
cluster_repr_train = X_train_final.iloc[:, :min(12, X_train_final.shape[1])].values
cluster_repr_test = X_test_final.iloc[:, :min(12, X_test_final.shape[1])].values

n_clusters = 8 if X_train_final.shape[0] > 10000 else 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_train = kmeans.fit_predict(cluster_repr_train)
cluster_labels_test = kmeans.predict(cluster_repr_test)

# Add cluster label columns as dummies
cluster_train_df = pd.get_dummies(cluster_labels_train, prefix="cluster_k")
cluster_test_df = pd.get_dummies(cluster_labels_test, prefix="cluster_k")
cluster_train_df, cluster_test_df = cluster_train_df.align(cluster_test_df, join='outer', axis=1, fill_value=0)

X_train_final = pd.concat([X_train_final.reset_index(drop=True), cluster_train_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_final.reset_index(drop=True), cluster_test_df.reset_index(drop=True)], axis=1)

print("   Shapes after clusters:", X_train_final.shape, X_test_final.shape)

# -------------------------
# 10) Scaling final features (fit on train only)
# -------------------------
print("\n⚖️ Escalando características finales (fit en train)")

final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_final)
X_test_scaled = final_scaler.transform(X_test_final)

joblib.dump(final_scaler, os.path.join(MODELS_DIR, "final_scaler.pkl"))
joblib.dump(pca_components_by_group, os.path.join(MODELS_DIR, "ipca_blocks.pkl"))
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_cluster.pkl"))
print("   ✅ Artifacts saved: final_scaler, ipca_blocks, kmeans_cluster")

# -------------------------
# 11) Training models
# -------------------------
print("\n🏋️ Entrenando modelos (robustos y optimizados)")

models_list = {}
models_list["Linear_Regression"] = LinearRegression()
models_list["Linear_SVR"] = LinearSVR(max_iter=5000)
models_list["RandomForest"] = RandomForestRegressor(n_estimators=RF_N_EST, n_jobs=-1, random_state=42)

if XGBRegressor is not None:
    models_list["XGBoost"] = XGBRegressor(n_estimators=XGB_N_EST if (XGB_N_EST:=XGB_N_EST if globals().get('XGB_N_EST') else XGB_N_EST) else XGB_N_EST, n_jobs=-1, verbosity=0)
if LGBMRegressor is not None:
    models_list["LightGBM"] = LGBMRegressor(n_estimators=LGB_N_EST, n_jobs=-1)

# Stacking (use RF + XGB if available)
stack_estimators = [('rf', RandomForestRegressor(n_estimators=80, random_state=42))]
if "XGBoost" in models_list:
    stack_estimators.append(('xgb', XGBRegressor(n_estimators=80, random_state=42, verbosity=0)))
stack = StackingRegressor(estimators=stack_estimators, final_estimator=LinearRegression(), n_jobs=-1)
models_list["Stacking"] = stack

# Voting ensemble of best candidates (choose those present)
voting_estimators = []
for k in ["RandomForest","XGBoost","LightGBM"]:
    if k in models_list:
        voting_estimators.append((k.lower(), models_list[k]))
if voting_estimators:
    models_list["Voting"] = VotingRegressor(estimators=voting_estimators, n_jobs=-1)

results = []
for name, model in models_list.items():
    print(f"\n🔹 Entrenando {name}")
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results.append({"Modelo": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"   ✅ Guardado: {model_path}")
        print(f"   ➤ MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    except Exception as e:
        print(f"   ❌ Error entrenando {name}: {e}")

# -------------------------
# 12) Results export
# -------------------------
if results:
    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(os.path.join(MODELS_DIR, "model_results.csv"), index=False)
    print("\n📊 Resultados guardados en models/model_results.csv")
    print(results_df)
else:
    print("\n❌ Ningún modelo entrenó correctamente. Revisa errores previos.")

# -------------------------
# 13) Save feature columns order
# -------------------------
feature_columns = list(X_train_final.columns)
joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))
print("✅ feature_columns.pkl guardado.")

print("\n🎯 Proceso completado. Artifacts en 'models/' y 'processed/'.")
