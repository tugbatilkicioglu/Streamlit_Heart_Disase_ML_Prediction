import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("heartfeature.csv")
#Define model file paths as constants
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression_best_model.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors_best_model.joblib",
    "Decision Tree": "decision_tree_best_model.joblib",
    "Random Forest": "random_forest_best_model.joblib",
    "Gradient Boosting": "gradient_boosting_best_model.joblib",
    "XGBoost": "xgboost_best_model.joblib",
}





st.set_page_config(
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)





# Models loading function
@st.cache_resource
def load_models():
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return models

# Prediction function
def predict(model, input_data):
    loaded_model = model
    prediction = loaded_model.predict(input_data)
    return prediction





# Homepage
def home():
    # Ana sayfa içeriği
    st.title("❤️ Happy Hearts: ML-Powered Heart Disease Prediction")
    st.markdown("Bulgu sonuçlarını 6 farklı makine öğrenmesi modeli ile analiz eder ve %98 doğruluk oranıyla cevap verir.")
    st.image("anasayfa.png", caption="")



def presentation():
    st.markdown(
        f'<iframe src="https://gamma.app/embed/rhes0yzwx5jwg5o" style="width: 100vw; height: 130vh; border: none;" allow="fullscreen" "></iframe>',
        unsafe_allow_html=True
    )


# Visualitazion Page
def visualization():
    st.subheader("Görselleştirme Sayfası")

    
    crosstab_df = pd.crosstab(df["sex"], df["target"])

    st.subheader("Heart Disease Frequency for Sex")
    st.bar_chart(crosstab_df, use_container_width=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    crosstab_df.plot(kind="bar", color=["salmon", "lightblue"], ax=ax)
    plt.title("Heart Disease Frequency for Sex")
    plt.xlabel("0 = No Disease, 1 = Disease")
    plt.ylabel("Amount")
    plt.legend(["Female", "Male"])
    plt.xticks(rotation=0)
    st.pyplot(fig)
    # plot_square(number)
    # age_range_thalach_box_plot(df)

    # Scatter plot görselleştirmesi
    st.subheader("Scatter Plot: Age Max Heart Rate Ratio vs. Target")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age_max_heart_rate_ratio', y='target', data=df, ax=ax)
    plt.title("Age Max Heart Rate Ratio vs. Target")
    plt.xlabel("Age Max Heart Rate Ratio")
    plt.ylabel("Target")
    plt.show()
    st.pyplot(fig)
    # feature_importance_bar_plot(feature_importances, feature_names)
    # roc_curve_plot(fpr, tpr)
    #######
    #######
    # Boxplot 
    st.subheader("Boxplot: Age vs. Thalach by Target")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='age', y='thalach', hue='target', data=df, ax=ax)
    plt.title("Age vs. Thalach by Target")
    plt.xlabel("Age")
    plt.ylabel("Thalach")
    plt.show()
    st.pyplot(fig)
    #######
    #######

    # Feature importance




def main():
    gif_path = "kalp.gif"

    # Sidebar'a GIF'i ekleyin
    st.sidebar.image(gif_path, use_column_width=True, caption="")

    # Sol tarafta menü oluşturma
    menu = ["Ana Sayfa", "Görselleştirme Sayfası", "Sunum Sayfası", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Ana Sayfa":
        home()

    elif choice == "Görselleştirme Sayfası":
        visualization()

    elif choice == "Sunum Sayfası":
        presentation()

    elif choice == "Predict":
        # Load model
        models = load_models()

        st.subheader("Predict")

        # Get data from user
        input_data, sex = get_user_input()

        # Model selection box
        selected_model = st.sidebar.selectbox("Select a model", list(MODEL_PATHS.keys()), key="model_selectbox")

        # Copy of first model
        initial_model = deepcopy(models[selected_model])

        # Variable to reset the prediction result
        prediction_result = None

        if st.button("Predict", key="predict_button"):
            # Prediction
            prediction = predict(models[selected_model], input_data)

            prediction_result = "KALP RAHATSIZLIĞI YOK 💖" if prediction[0] == 0 else "KALP RAHATISZLIĞI VAR 💔"

            # Printing the prediction result to the screen
            if prediction_result == "KALP RAHATISZLIĞI VAR 💔":
                st.error(f"Kullanılan Model {selected_model}: {prediction_result}")
                # Show image by gender
                if sex == "Kadın":
                    st.image("kadin.jpg", caption="", use_column_width=False)
                elif sex == "Erkek":
                    st.image("erkek.jpg", caption="", use_column_width=False)
            else:
                st.success(f"Kullanılan Model {selected_model}: {prediction_result}")

                #Show heart icon and add balloons in case of No Heart Disease
                if prediction_result == "KALP RAHATSIZLIĞI YOK 💖":
                    heart_image_path = "health.jpg"  # Kalp simgesinin gerçek yolunu belirtin
                    if os.path.exists(heart_image_path):
                        st.image(heart_image_path, caption="", use_column_width=False)
                        st.balloons()
                    else:
                        st.warning("Warning: Heart image not found at the specified path.")

            # Reset model and prediction result
            models[selected_model] = initial_model
            prediction_result = None

# ... Others

def get_user_input():
    age = st.slider("Yaş:", min_value=29, max_value=79, value=40)
    sex = st.radio("Cinsiyet:", options=["Kadın", "Erkek"])
    sex_Male = 1 if sex == "Erkek" else 0
    cp = st.slider("Göğüs Ağrısı Tipleri (0 : Kan akışının azalmasından kaynaklı - 1 : Kalbe bağlı olmayan ağrı - 2 : Yemek borusu spazmından kaynaklı  3 : Hastalık belirtisi göstermeyen ağrı):", min_value=0, max_value=3, value=1)
    trestbps = st.slider("İstirahat Kan Basıncı (mm Hg):", min_value=90, max_value=200, value=120)
    chol = st.slider("Kolesterol (mg/dl):", min_value=50, max_value=600, value=200)
    fbs = st.radio("Açlık Kan Şekeri (> 120 mg/dl):", options=["Hayır", "Evet"])
    fbs = 1 if fbs == "Evet" else 0
    restecg = st.slider("İstirahat EKG Sonuçlar (0 : Normal - 1 : Orta - 2 : Yüksek):", min_value=0, max_value=2, value=1)
    thalach = st.slider("Maksimum Kalp Atış Hızı:", min_value=70, max_value=220, value=150)
    exang = st.radio("Egzersize Bağlı Göğüs Ağrısı (0 : Göğüs ağrısı yok - 1 : Göğüs ağrısı var):", options=["Hayır", "Evet"])
    exang = 1 if exang == "Evet" else 0
    oldpeak = st.slider("Egzersizle ST Depresyonu (İskemi; kan akışının zayıflaması):", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.slider("Egzersiz EKG esnasında ST Segmentinin Eğimi (0 : Eğim Düz - 1 : Eğim yavaş artan - 2 : Eğim hızlı artan):", min_value=0, max_value=2, value=1)
    ca = st.slider(" Damarlardaki Kalsiyum Birikimi (0 : Yok - 1 : Çok az - 2 : Orta - 3 : yüksek ):", min_value=0, max_value=3, value=0)
    thal = st.slider("Thalassemi, kandaki hemoglobin proteini etkilenme (1 : Az - 2 : Orta - 3 : Yüksek)):", min_value=1, max_value=3, value=2)

    # Convert the user's input to the appropriate format for the model
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_Male],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],
        'age_max_heart_rate_ratio': [age / thalach],
        'cholesterol_hdl_ratio': [chol / thalach],
        'heart_rate_reserve': [thalach - trestbps]
    })
    return input_data, sex

# Uygulamayı başlat
if __name__ == '__main__':
    models = load_models()
    main()
