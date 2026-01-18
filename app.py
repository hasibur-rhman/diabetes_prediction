import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("diabetes_model.pkl","rb") as f:
    model=pickle.load(f)

def predict_diabetes(preg,glu,bp,skin,ins,bmi,dpf,age):

    bf=bmi*skin
    mi=glu*ins

    input_df=pd.DataFrame([[
        preg,glu,bp,skin,ins,bmi,dpf,age,bf,mi
    ]],
    columns=[
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age",
        "Body_Fat_Index","Metabolic_Index"
    ])

    prediction=model.predict(input_df)[0]
    

    if prediction==1:
        result="High Risk of Diabetes"
    else:
        result="Low Risk of Diabetes"

    return f"Prediction: {result}"

inputs=[
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="Blood Pressure"),
    gr.Number(label="Skin Thickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="Diabetes Pedigree Function"),
    gr.Number(label="Age")
]

app=gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction System"
)

app.launch(share=True)
