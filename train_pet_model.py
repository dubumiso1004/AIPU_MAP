import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 데이터 경로
data_path = r"C:\Users\jsj54\OneDrive\Desktop\AIPU_MAP\total_svf_gvi_bvi_250613.xlsx"

# 데이터 불러오기
df = pd.read_excel(data_path)

# 입력(X)과 출력(y) 설정
X = df[['SVF', 'GVI', 'BVI', 'AirTemperature', 'Humidity', 'WindSpeed']]
y = df['PET']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 모델 저장 경로
model_path = r"C:\Users\jsj54\OneDrive\Desktop\AIPU_MAP\pet_rf_model_trained.pkl"
joblib.dump(model, model_path)

print(f"✅ 모델이 성공적으로 저장되었습니다: {model_path}")
