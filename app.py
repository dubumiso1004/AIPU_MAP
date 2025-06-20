import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
import joblib

# ✅ 위경도 DMS → 소수점 변환 함수
def dms_to_decimal(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# ✅ 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")
    df.columns = df.columns.str.strip().str.lower().str.replace('\r', '').str.replace('\n', '')
    df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
    df["lon_decimal"] = df["lon"].apply(dms_to_decimal)
    return df

# ✅ 모델 로딩
model = joblib.load("pet_rf_model_trained.pkl")
df = load_data()

# ✅ 앱 UI 구성
st.set_page_config(page_title="AI PET 예측 (기상청 JSON)", layout="wide")
st.title("📍 실측값 + 기상청 JSON 기반 AI PET 예측")
st.caption("측정된 SVF/GVI/BVI + kma_latest_weather.json 기상 데이터를 기반으로 PET를 예측합니다.")

# ✅ 지도 클릭 UI (좌우 정렬)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 🗺️ 지도에서 위치 선택")
    map_center = [35.233, 129.08]
    m = folium.Map(location=map_center, zoom_start=17)
    click_data = st_folium(m, height=450)

with col2:
    if click_data and click_data["last_clicked"]:
        lat = click_data["last_clicked"]["lat"]
        lon = click_data["last_clicked"]["lng"]

        st.markdown("### 📌 선택한 위치")
        st.write(f"위도: {lat:.5f}, 경도: {lon:.5f}")

        try:
            # ✅ 클릭 위치와 가장 가까운 측정지점 탐색
            df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)**0.5
            nearest = df.loc[df["distance"].idxmin()]
        except Exception as e:
            st.error(f"❌ 측정지점 탐색 실패: {e}")
            st.stop()

        # ✅ 측정값 출력
        st.markdown("#### ✅ 측정된 시각 지표 (SVF, GVI, BVI)")
        st.write({
            "측정지점": nearest["location_name"],
            "SVF": nearest["svf"],
            "GVI": nearest["gvi"],
            "BVI": nearest["bvi"]
        })

        try:
            # ✅ 기상청 JSON 파일 로드 (사전 저장된 최신 JSON)
            with open("kma_latest_weather.json", "r", encoding="utf-8") as f:
                weather = json.load(f)

            air_temp = weather["airtemperature"]
            humidity = weather["humidity"]
            wind_speed = weather["windspeed"]

            st.markdown("#### 🌤 기상청 실시간 기상 (JSON 기반)")
            st.write({
                "기온 (°C)": air_temp,
                "습도 (%)": humidity,
                "풍속 (m/s)": wind_speed
            })

        except Exception as e:
            st.error(f"기상청 JSON 불러오기 실패: {e}")
            st.stop()

        # ✅ AI 입력값 구성 + 예측
        X_input = pd.DataFrame([{
            "SVF": nearest["svf"],
            "GVI": nearest["gvi"],
            "BVI": nearest["bvi"],
            "AirTemperature": air_temp,
            "Humidity": humidity,
            "WindSpeed": wind_speed
        }])
        predicted_pet = model.predict(X_input)[0]

        # ✅ 예측 결과 출력
        st.markdown("#### 🤖 AI 기반 PET 예측 결과")
        st.success(f"예측 PET: **{predicted_pet:.2f}°C**")
        st.caption("※ 기상청 JSON (kma_latest_weather.json)을 기반으로 예측되었습니다.")
    else:
        st.info("지도를 클릭하여 예측을 시작하세요.")
