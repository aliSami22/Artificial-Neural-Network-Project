import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from minisom import MiniSom
import streamlit as st

# تحميل البيانات
df = pd.read_csv('Egypt_Houses_Price.csv')

# تحديد الأعمدة المهمة
columns_to_use = ['Type', 'Price', 'Bedrooms', 'Bathrooms', 'Area', 'Furnished', 'Level', 'Compound', 'Payment_Option', 'Delivery_Term', 'City']
df = df[columns_to_use].dropna()

# تشفير البيانات النصية
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# تطبيع البيانات
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.drop('Price', axis=1))

# تدريب SOM
som = MiniSom(10, 10, data_scaled.shape[1], sigma=0.5, learning_rate=0.5)
som.random_weights_init(data_scaled)
som.train_random(data_scaled, 100)

# واجهة Streamlit
st.title("تحليل أسعار العقارات في مصر باستخدام SOM")

# مدخلات المستخدم
st.sidebar.header("أدخل خصائص العقار")

def user_input():
    input_data = {}
    for col in df.drop('Price', axis=1).columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            input_data[col] = st.sidebar.selectbox(col, options)
        else:
            input_data[col] = st.sidebar.number_input(col, min_value=0)
    return input_data

user_vals = user_input()

# تحويل المدخلات
input_df = pd.DataFrame([user_vals])
for col, le in label_encoders.items():
    val = input_df[col].values[0]
    input_df[col] = le.transform([val])


input_scaled = scaler.transform(input_df)

# إيجاد أقرب عقدة
winner = som.winner(input_scaled[0])
similar_indexes = [i for i, x in enumerate(data_scaled) if som.winner(x) == winner]

# عرض النتائج
st.subheader("عقارات مشابهة")
similar_properties = df.iloc[similar_indexes]
similar_properties['Price'] = df['Price'].iloc[similar_indexes].values
st.dataframe(similar_properties.head(10))

# السعر المتوقع
predicted_price = similar_properties['Price'].mean()
st.subheader(f"السعر المتوقع: {predicted_price:,.0f} جنيه")
