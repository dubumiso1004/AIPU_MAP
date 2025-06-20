import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import joblib

# DMS → Decimal Degrees
def dms_to_decimal(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# 측정 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")
    df.columns = df.columns.str.strip().str.lower().str.replace('\r', '').str.replace('\n', '')
    df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
    df["lon_decimal"] = df["lon"].apply(dms_to_decimal)
    return df

# 모델 로딩
model = joblib.load("pet_rf_model_trained.pkl")
df = load_data()

# Streamlit UI
st.set_page_config(page_title="AI PET 예측 + 조절", layout="wide")
st.title("📍 실측값 + 실시간 기상 기반 PET 예측")
st.caption("SVF, GVI, BVI를 조절하며 PET 예측값 변화를 확인할 수 있습니다.")

# 지도와 결과 분리
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
            df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)**0.5
            nearest = df.loc[df["distance"].idxmin()]
        except Exception as e:
            st.error(f"❌ 측정지점 탐색 실패: {e}")
            st.stop()

        # SVF/GVI/BVI 슬라이더 조절
        st.markdown("#### 🎛️ 시뮬레이션: 시각지표 조절")
        svf = st.slider("SVF (하늘 비율)", 0.0, 1.0, float(nearest["svf"]), 0.01)
        gvi = st.slider("GVI (녹지 비율)", 0.0, 1.0, float(nearest["gvi"]), 0.01)
        bvi = st.slider("BVI (건물 비율)", 0.0, 1.0, float(nearest["bvi"]), 0.01)

        # OpenWeatherMap API 호출
        try:
            api_key = "2ced117aca9b446ae43cf82401d542a8"  # ← 본인 API 키
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url)
            weather = response.json()

            if "main" in weather and "wind" in weather:
                air_temp = weather["main"]["temp"]
                humidity = weather["main"]["humidity"]
                wind_speed = weather["wind"]["speed"]

                st.markdown("#### 🌤 실시간 기상 정보")
                st.write({
                    "기온 (°C)": air_temp,
                    "습도 (%)": humidity,
                    "풍속 (m/s)": wind_speed
                })
            else:
                raise Exception(weather.get("message", "기상 정보 없음"))

        except Exception as e:
            st.warning(f"⚠️ 실시간 기상 실패 → 측정값 사용\n({e})")
            air_temp = nearest["airtemperature"]
            humidity = nearest["humidity"]
            wind_speed = nearest["windspeed"]

        # AI 예측
        X_input = pd.DataFrame([{
            "SVF": svf,
            "GVI": gvi,
            "BVI": bvi,
            "AirTemperature": air_temp,
            "Humidity": humidity,
            "WindSpeed": wind_speed
        }])
        predicted_pet = model.predict(X_input)[0]

        st.markdown("#### 🤖 AI 기반 PET 예측 결과")
        st.success(f"예측 PET: **{predicted_pet:.2f}°C**")
        st.caption("측정값 + 실시간 기상 데이터 + 조절된 시각 지표를 기반으로 예측된 결과입니다.")

    else:
        st.info("지도를 클릭해 위치를 선택해주세요.")
