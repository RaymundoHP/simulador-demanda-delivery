import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

class PredictionPipeline:
    def __init__(self, models_dir="models"):
        # Configuración de rutas robusta
        self.models_dir = Path(__file__).parent / models_dir if not os.path.isabs(models_dir) else Path(models_dir)
        self.model = None
        self.scaler = None
        self.features = None
        self.ipca_blocks = None
        self.kmeans = None
        self.is_loaded = False
        
        self._load_resources()

    def _load_resources(self):
        """Carga todos los artefactos del modelo."""
        print("🔄 Iniciando carga de recursos de IA...")
        try:
            # 1. Cargar Estructura de Datos
            self.features = joblib.load(self.models_dir / "feature_columns.pkl")
            self.scaler = joblib.load(self.models_dir / "final_scaler.pkl")
            
            try: self.ipca_blocks = joblib.load(self.models_dir / "ipca_blocks.pkl")
            except: self.ipca_blocks = None
            
            try: self.kmeans = joblib.load(self.models_dir / "kmeans_cluster.pkl")
            except: self.kmeans = None

            # 2. Cargar Modelo
            priority_list = ["Stacking.pkl", "Voting.pkl", "LightGBM.pkl", "RandomForest.pkl", "Linear_Regression.pkl"]
            
            for name in priority_list:
                path = self.models_dir / name
                if path.exists():
                    self.model = joblib.load(path)
                    print(f"✅ Modelo activo: {name}")
                    self.is_loaded = True
                    break
            
            if not self.is_loaded:
                print("⚠️ No se encontraron modelos .pkl válidos.")

        except Exception as e:
            print(f"❌ Error Crítico cargando modelos: {e}")
            self.is_loaded = False

    def _create_base_vector(self):
        """Crea un DataFrame base lleno de ceros."""
        return pd.DataFrame(0.0, index=[0], columns=self.features)

    def predict(self, user_inputs):
        """Pipeline de Inferencia v6.0"""
        if not self.is_loaded:
            return 0.0

        try:
            X_raw = self._create_base_vector()

            # --- 1. PERFIL DE COCINA Y CONTEXTO ---
            cuisine = user_inputs.get('cuisine', 'generic')
            event = user_inputs.get('event', 'none')
            weather = user_inputs.get('weather', 'sunny')
            temp = user_inputs.get('temp', 24)
            trend = user_inputs.get('trend', 'stable')
            
            cuisine_factor = 1.0
            
            if cuisine == 'fast_food':
                if event == 'match': cuisine_factor *= 1.15
                if weather in ['rain', 'storm']: cuisine_factor *= 1.10
            elif cuisine == 'healthy':
                if event == 'match': cuisine_factor *= 0.85
                if weather == 'storm': cuisine_factor *= 0.95 
            elif cuisine == 'beverages':
                if temp > 28: cuisine_factor *= 1.20
                if temp < 15: cuisine_factor *= 0.80

            trend_mult = 1.0
            if trend == 'growing': trend_mult = 1.15
            elif trend == 'declining': trend_mult = 0.85

            weather_bonus = 1.0
            if weather in ['rain', 'storm']: weather_bonus = 1.20
            elif weather == 'cloudy': weather_bonus = 1.05
            
            context_factor = trend_mult * weather_bonus * cuisine_factor

            # --- 2. CÁLCULOS PREVIOS ---
            radius = user_inputs.get('radius', 2.0)
            calculated_area = 3.1416 * (radius ** 2)

            base_p = user_inputs.get('base_price', 150)
            disc_percent = user_inputs.get('discount', 0)
            checkout_p = base_p * (1 - (disc_percent / 100.0))
            disc_rate = disc_percent / 100.0

            # --- 3. INYECCIÓN DE DATOS ---
            if 'orders' in user_inputs:
                val = user_inputs['orders'] * context_factor
                for col in ['food_demand_v2_clean_num_orders_mean', 'food_demand_v2_clean_num_orders_max', 'food_demand_v2_clean_num_orders_min']:
                    if col in self.features: X_raw[col] = val

            for col in ['base_price', 'base__base_price', 'food_demand_v2_clean_base_price_mean']:
                if col in self.features: X_raw[col] = base_p
            for col in ['checkout_price', 'checkout__checkout_price', 'food_demand_v2_clean_checkout_price_mean']:
                if col in self.features: X_raw[col] = checkout_p
            for col in ['discount_rate', 'discount__discount_rate', 'food_demand_v2_clean_discount_rate_mean']:
                if col in self.features: X_raw[col] = disc_rate

            temp_val = temp
            if weather == 'rain': temp_val -= 2 
            if weather == 'storm': temp_val -= 5
            for col in ['city_temperature_AvgTemperature_mean', 'city_temperature_Month_mean']:
                if col in self.features: X_raw[col] = temp_val

            for col in ['op_area', 'op__op_area', 'food_demand_v2_clean_op_area_mean']:
                if col in self.features: X_raw[col] = calculated_area

            if user_inputs.get('promo_email'):
                for col in ['emailer_for_promotion', 'food_demand_v2_clean_emailer_for_promotion_mean']:
                    if col in self.features: X_raw[col] = 1.0
            if user_inputs.get('promo_homepage'):
                for col in ['homepage_featured', 'food_demand_v2_clean_homepage_featured_mean', 'homepage__homepage_featured']:
                    if col in self.features: X_raw[col] = 1.0
            if user_inputs.get('promo_email') or user_inputs.get('promo_homepage'):
                for col in ['promo_active', 'promo__promo_active']:
                    if col in self.features: X_raw[col] = 1.0

            # --- 4. TRANSFORMACIONES ---
            if self.ipca_blocks:
                for group_name, block_data in self.ipca_blocks.items():
                    try:
                        cols = block_data['cols']
                        ipca = block_data['ipca']
                        scaler_g = block_data['scaler']
                        sub_df = X_raw.reindex(columns=cols, fill_value=0)
                        vect_pca = ipca.transform(scaler_g.transform(sub_df))
                        for i in range(vect_pca.shape[1]):
                            pc_col = f"{group_name}_PC{i+1}"
                            if pc_col in self.features: X_raw[pc_col] = vect_pca[0, i]
                    except: pass

            if self.kmeans:
                try:
                    n_cols = self.kmeans.n_features_in_
                    cluster_id = self.kmeans.predict(X_raw.iloc[:, :n_cols])[0]
                    cluster_col = f"cluster_k_{cluster_id}"
                    if cluster_col in self.features: X_raw[cluster_col] = 1.0
                except: pass

            # --- 5. PREDICCIÓN FINAL ---
            X_final = X_raw[self.features]
            X_scaled = self.scaler.transform(X_final)
            
            raw_prediction = self.model.predict(X_scaled)[0]
            
            event_factor = 1.0
            if event == 'traffic': event_factor = 0.90
            
            final_prediction = raw_prediction * event_factor
            return max(0, final_prediction)

        except Exception as e:
            print(f"⚠️ Error en inferencia: {e}")
            return user_inputs.get('orders', 0)