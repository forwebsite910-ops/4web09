from pycaret.regression import load_model, predict_model # type: ignore
import streamlit as st
import pandas as pd
from PIL import Image

model=load_model('dt_insurance_charges')

def predict(model,input_df):
    predictions_df=predict_model(estimator=model,data=input_df)
    predictions=predictions_df.iloc[0]['prediction_label']
    return predictions

def run():
    image=Image.open('image.png')
    image_hospital=Image.open('image.jpg')

    st.image(image)
    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.image(image_hospital)
    add_selectbox=st.sidebar.selectbox("How would you like to predict?",("Online","Batch"))
    st.title("Insurance charges prediction app")

    if add_selectbox=='Online':
        age=st.number_input('Age',min_value=1,max_value=100,value=25)
        sex=st.selectbox('Sex',['Male','Female'])
        bmi=st.number_input('BMI',min_value=10,max_value=50,value=10)
        children=st.select_box('Children',[0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker='yes'
        else:
            smoker='no'
        region=st.selectbox('Region',['Southwest','Northwest','Northeast','Southeast'])
        output=""


        input_dict={'Age':age,'Sex':sex,'BMI':bmi,'Children':children,'Smoker':smoker,'Region':region}
        input_df=pd.DataFrame([input_dict])

        if st.button("Predict"):
            output=predict(model=model, input_df=input_df)
            output="$"+str(output)
            st.success('This Output is {}'.format(output))

    if add_selectbox=='Batch':
        file_upload= st.file_uploader("Drop files here for predictions",type=[])

            
                


if __name__=='_main_':
    run()

