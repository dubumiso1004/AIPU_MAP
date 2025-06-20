import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. 데이터 불러오기
df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx")

# 2. 입력 변수(X), 타깃 변수(y) 설정
X = df[['SVF', 'GVI', 'BVI', 'AirTemperature', 'Humidity', 'WindSpeed']]
y = df['PET']

# 3. 모델 정의 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. 모델 저장
joblib.dump(model, 'pet_rf_model_trained.pkl')
print("✅ 모델이 'pet_rf_model_trained.pkl' 이름으로 저장되었습니다.")
