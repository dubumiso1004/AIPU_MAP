import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import joblib

# ✅ DMS 형식 위경도 → 소수점 변환
def dms_to_decimal(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# ✅ 사용자 측정 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")
    df.columns = df.columns.str.strip().str.lower().str.replace('\r', '').str.replace('\n', '')
    df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
    df["lon_decimal"] = df["lon"].apply(dms_to_decimal)
    return df

# ✅ AI 모델 로드
model = joblib.load("pet_rf_model_trained.pkl")

# ✅ Streamlit UI
st.set_page_config(page_title="AI 기반 PET 예측", layout="centered")
st.title("📍 사용자 측정값 + 실시간 기상 기반 PET 예측")
st.caption("측정된 SVF, GVI, BVI + OpenWeatherMap 실시간 기상 데이터 기반 예측")

# ✅ 데이터 로딩
df = load_data()

# ✅ 지도 표시 및 클릭 이벤트
map_center = [35.233, 129.08]
m = folium.Map(location=map_center, zoom_start=17)
click_data = st_folium(m, height=450)

# ✅ 클릭 처리
if click_data and click_data["last_clicked"]:
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]

    st.subheader("🔎 선택 위치")
    st.write(f"위도: {lat:.5f}, 경도: {lon:.5f}")

    try:
        df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)**0.5
        nearest = df.loc[df["distance"].idxmin()]
    except Exception as e:
        st.error(f"❌ 측정 위치 탐색 실패: {e}")
        st.stop()

    # ✅ 측정값 표시
    st.markdown("### 📌 측정값 (SVF, GVI, BVI)")
    st.write({
        "지점명": nearest["location_name"],
        "SVF": nearest["svf"],
        "GVI": nearest["gvi"],
        "BVI": nearest["bvi"]
    })

    # ✅ OpenWeatherMap 실시간 기상 데이터
    try:
        api_key = "2ced117aca9b446ae43cf82401d542a8"  # ← 당신이 제공한 키
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(weather_url)
        weather = response.json()

        if "main" in weather and "wind" in weather:
            air_temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]

            st.markdown("### 🌤 실시간 기상 정보 (OpenWeatherMap)")
            st.write({
                "기온 (°C)": air_temp,
                "습도 (%)": humidity,
                "풍속 (m/s)": wind_speed
            })
        else:
            raise Exception(weather.get("message", "기상 정보 없음"))

    except Exception as e:
        st.warning(f"❌ 실시간 기상 불러오기 실패 → 측정값 사용\n({e})")
        air_temp = nearest["airtemperature"]
        humidity = nearest["humidity"]
        wind_speed = nearest["windspeed"]

    # ✅ AI 예측
    X_input = pd.DataFrame([{
        "SVF": nearest["svf"],
        "GVI": nearest["gvi"],
        "BVI": nearest["bvi"],
        "AirTemperature": air_temp,
        "Humidity": humidity,
        "WindSpeed": wind_speed
    }])

    predicted_pet = model.predict(X_input)[0]

    # ✅ 결과 표시
    st.markdown("### 🤖 AI 기반 PET 예측 결과")
    st.success(f"예측 PET: **{predicted_pet:.2f}°C**")
    st.caption("이 예측은 RandomForest 머신러닝 모델을 기반으로 생성되었습니다.")

else:
    st.info("지도를 클릭해 PET 예측을 시작하세요.")
