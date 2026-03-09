import streamlit as st
import pandas as pd
import requests
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Налаштування сторінки
st.set_page_config(page_title="Прогноз опадів ML", layout="wide")
st.title("🌦 Прогноз опадів за допомогою Machine Learning")


# Функція для отримання даних з Open-Meteo
def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "precipitation_sum", "rain_sum"],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['daily'])
        return df
    else:
        st.error(f"Помилка запиту до API: {response.status_code}")
        return None


# Секція 1: Завантаження даних
st.header("1. Отримання даних з Open-Meteo")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Широта (Latitude)", value=50.4501)  # Київ
with col2:
    lon = st.number_input("Довгота (Longitude)", value=30.5234)  # Київ

# Беремо дані за останній рік для достатньої вибірки
default_end = datetime.date.today() - datetime.timedelta(days=1)
default_start = default_end - datetime.timedelta(days=365)

date_range = st.date_input("Період даних", value=(default_start, default_end))

if st.button("Отримати дані з Open-Meteo"):
    if len(date_range) == 2:
        with st.spinner('Завантаження даних...'):
            df = fetch_weather_data(lat, lon, date_range[0], date_range[1])
            if df is not None:
                # Зберігаємо у CSV
                df.to_csv("weather_daily.csv", index=False)
                st.success("Дані успішно завантажено та збережено у 'weather_daily.csv'!")
                st.session_state['df'] = df
                st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Будь ласка, оберіть коректний діапазон дат.")

# Секція 2: Навчання моделі
st.header("2. Навчання ML-моделі")

if st.button("Навчити модель"):
    if 'df' not in st.session_state:
        try:
            st.session_state['df'] = pd.read_csv("weather_daily.csv")
            st.info("Дані завантажено з локального файлу weather_daily.csv")
        except FileNotFoundError:
            st.error("Дані не знайдено. Спочатку завантажте їх на кроці 1.")
            st.stop()

    df = st.session_state['df'].copy()

    # Підготовка даних: ознаки
    features = ['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'precipitation_sum', 'rain_sum']

    # Формування цільової змінної (опади: так/ні). 
    # Робимо зсув на -1, щоб за сьогоднішніми ознаками передбачати завтрашні опади
    df['target'] = (df['precipitation_sum'] > 0).astype(int)
    df['target'] = df['target'].shift(-1)

    # Зберігаємо останній рядок для прогнозу на завтра (там target = NaN)
    last_day_data = df.iloc[-1:]
    st.session_state['last_day_data'] = last_day_data[features]
    st.session_state['last_date'] = last_day_data['time'].values[0]

    # Видаляємо останній рядок, бо для нього немає правильної відповіді (target)
    df = df.dropna()

    X = df[features]
    y = df['target']

    # Розбиття на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Навчання моделі
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Прогноз на тестовій вибірці
    y_pred = model.predict(X_test)

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.session_state['model'] = model
    st.session_state['is_trained'] = True

    st.success("Модель успішно навчено!")
    st.write("### Метрики оцінки точності:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")

# Секція 3: Прогноз
st.header("3. Прогноз опадів")

if st.button("Зробити прогноз"):
    if 'model' not in st.session_state or not st.session_state.get('is_trained'):
        st.warning("Спочатку навчіть модель на кроці 2!")
    else:
        model = st.session_state['model']
        X_pred = st.session_state['last_day_data']
        last_date = st.session_state['last_date']

        # Обчислення прогнозу та ймовірності
        prediction = model.predict(X_pred)[0]
        probabilities = model.predict_proba(X_pred)[0]
        prob_rain = probabilities[1] * 100

        st.write(f"Прогноз на день, наступний після **{last_date}**:")

        if prediction == 1:
            st.error("🌧 **Очікуються опади**")
        else:
            st.success("☀️ **Опадів не очікується**")

        st.info(f"Ймовірність опадів: **{prob_rain:.1f}%**")