import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# تحميل البيانات
df = pd.read_csv("Egypt_Houses_Price.csv")

# إسقاط أي أعمدة غير مفيدة
df.dropna(inplace=True)

# حدد الأعمدة المستخدمة في التدريب
features = ['rooms', 'area']
target = 'price'

# التأكد من أن الأعمدة موجودة
if not all(col in df.columns for col in features + [target]):
    st.error("البيانات لا تحتوي على الأعمدة المطلوبة.")
    st.stop()

# تجهيز البيانات
X = df[features].values
y = df[target].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# تدريب SOM
som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 100)

# واجهة Streamlit
st.title("تحليل أسعار العقارات في مصر باستخدام SOM")

st.header("أدخل مواصفات العقار الجديد:")
room_input = st.number_input("عدد الغرف", min_value=1, step=1)
area_input = st.number_input("المساحة (متر مربع)", min_value=20, step=10)

if st.button("تقدير السعر و عرض أقرب حالة"):

    # تجهيز الإدخال
    input_data = scaler.transform([[room_input, area_input]])

    # الحصول على الخلية الأقرب في SOM
    winner = som.winner(input_data[0])

    # إيجاد أقرب نقطة في الداتا
    distances = []
    for i, x in enumerate(X_scaled):
        if som.winner(x) == winner:
            dist = np.linalg.norm(x - input_data[0])
            distances.append((dist, i))

    if not distances:
        st.warning("لا توجد حالة مشابهة في الخريطة.")
    else:
        # أقرب نقطة
        _, best_index = sorted(distances)[0]
        predicted_price = y[best_index]

        st.success(f"🔮 السعر المتوقع: {predicted_price:,.2f} جنيه مصري")

        st.subheader("📌 أقرب حالة مشابهة:")
        st.write(df.iloc[best_index])
