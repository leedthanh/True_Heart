
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn
import joblib as joblib


#DATASET_PATH = "heart_2020_cleaned.csv"
DATASET_PATH = "output_chunk_part_2.csv"
# LOG_MODEL_PATH = "logistic_regression.pkl"  #WORKS
LOG_MODEL_PATH = "regression_resampled.pkl"  #WORKS BUT LOW PRECISION
# LOG_MODEL_PATH = "svm_model.joblib"  #NOT WORKING CHECK INDEX INPUT
# LOG_MODEL_PATH = "support_vector_machine.pkl"  WORKS BUT PRECISION IS BAD
# LOG_MODEL_PATH = "neural_network_model.h5"   #not working 
# LOG_MODEL_PATH = "regression.pkl" NOT WORKING MISSING COLUMNS ERROR
def main():
    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH)
        heart_df = heart_df.to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df


    def user_input_features() -> pd.DataFrame:
        race = st.selectbox("Race", options=(race for race in heart.Race.unique()))
        sex = st.selectbox("Sex", options=(sex for sex in heart.Sex.unique()))
        age_cat = st.selectbox("Age category", options=(age_cat for age_cat in heart.AgeCategory.unique()))
        bmi_cat = st.selectbox("BMI category", options=(bmi_cat for bmi_cat in heart.BMICategory.unique()))
        sleep_time = st.number_input("How many hours on average do you sleep?", 0, 24, 7)
        gen_health = st.selectbox("How can you define your general health?",
                                options=(gen_health for gen_health in heart.GenHealth.unique()))
        phys_health = st.number_input("For how many days during the past 30 days was your physical health not good?", 0, 30, 0)
        ment_health = st.number_input("For how many days during the past 30 days was your mental health not good?", 0, 30, 0)
        phys_act = st.selectbox("Have you played any sports (running, biking, etc.) in the past month?", options=("No", "Yes"))
        smoking = st.selectbox("Have you smoked at least 100 cigarettes in your entire life (approx. 5 packs)?", options=("No", "Yes"))
        alcohol_drink = st.selectbox("Do you have more than 14 drinks of alcohol (men) or more than 7 (women) in a week?", options=("No", "Yes"))
        stroke = st.selectbox("Did you have a stroke?", options=("No", "Yes"))
        diff_walk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", options=("No", "Yes"))
        diabetic = st.selectbox("Have you ever had diabetes?", options=(diabetic for diabetic in heart.Diabetic.unique()))
        asthma = st.selectbox("Do you have asthma?", options=("No", "Yes"))
        kid_dis = st.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        skin_canc = st.selectbox("Do you have skin cancer?", options=("No", "Yes"))

        features = {
        'BMICategory':[bmi_cat],
        'Smoking':[smoking],
        'AlcoholDrinking':[alcohol_drink],
        'Stroke':[stroke],
        'DiffWalking':[diff_walk],
        'Sex':[sex],
        'AgeCategory':[age_cat],
        'Race':[race],
        'Diabetic':[diabetic],
        'PhysicalActivity':[phys_act],
        'GenHealth':[gen_health],
        'Asthma':[asthma],
        'KidneyDisease':[kid_dis],
        'SkinCancer':[skin_canc],
        'PhysicalHealth':[phys_health],
        'MentalHealth':[ment_health],
        'SleepTime':[sleep_time],
    }

        features_df = pd.DataFrame(features)

        return features_df

     
    st.set_page_config(
        page_title="Heart Disease Prediction App",
        )

    st.title("Heart Disease Prediction")
    st.subheader("True-Heart is a Machine learning model predicting heart disease.  Trained on dataset from the CDC.")
    st.write("Data source https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease")
    st.write("Please answer a set of questions that assess your risk of heart disease using logistic regression algorithm.")
    heart = load_dataset()

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

 

    df = df[:1]
    df.fillna(0, inplace=True)


    #convert to float 
    df = df.astype(float)


    log_model = joblib.load(open(LOG_MODEL_PATH, "rb"))
    
    # log_model = tf.keras.models.load_model(LOG_MODEL_PATH)

    submit = st.button("Make Prediction")

    if submit:

        prediction = log_model.predict(df)

        # Assuming the model predicts probabilities for a binary classification
        # predicted_class = np.argmax(prediction_prob)  # Get the index of the maximum probability

        if prediction[0] == 1:
            st.warning("The model predicts HIGH RISK.")
            st.warning("High Risk Detected! Take appropriate action.")
            st.write("CDC heart disease information: "
             "https://www.cdc.gov/heartdisease/index.htm")
            st.warning("Bidi bidi bom bom! I am not a doctor.  Go get a check up")
         
        else:
            st.success("The model predicts LOW RISK.")
            st.write("CDC heart disease information: "
         "https://www.cdc.gov/heartdisease/index.htm")
            st.success("Bidi bidi bom bom! I am not a doctor.  Go get a check up")
      
            st.balloons()
    
    st.write("Disclaimer: The results of this assessment are based on a machine learning model using logistic regression and are not a substitute for professional medical advice. Consult with a healthcare professional for accurate and personalized guidance.")


if __name__ == "__main__":
    main()
