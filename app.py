import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open('model.pkl', 'rb'))
gender_encoder = pickle.load(open('gender_encoder.pkl', 'rb'))
status_encoder = pickle.load(open('status_encoder.pkl', 'rb'))
student_data = pd.read_csv('encoded_student_data.csv')

st.title("🎓 Student Status Predictor")
st.write("Enter the student rollno to predict whether they will **Pass** or **Fail**.")

rollno= st.text_input("Enter Student Roll No").lower()

if st.button("Predict"):
    match = student_data[student_data["Roll.no"].str.lower() == rollno]

    if match.empty:
        st.error("❌ Student not found")
    else:
        
        decoded_gender = gender_encoder.inverse_transform(match['Gender'].values)
        display_data = match.copy()
        display_data['Gender'] = decoded_gender

        st.subheader("Student Details")
        st.dataframe(display_data[["Roll.no", "Name", "Percentage", "Gender"]])

        student_features = match[["Percentage", "Gender"]]
        prediction = model.predict(student_features)
        

        st.subheader("Status")
        if prediction == 0:
            st.error("📉 Status: Fail")
        else:
            st.success("🎉 Status: Pass")
