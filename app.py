import streamlit as st
import pickle 
import numpy as np

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
st.title('Laptop Price Predictor')

Company=st.selectbox('Brand',df['Company'].unique())
TypeName=st.selectbox('Type',df['TypeName'].unique())
Ram=st.selectbox('Ram in GB',[2,4,6,8,12,16,24,32,64])
weight=st.number_input('Weight of Laptop')
Touchscreen=st.selectbox('Touchscreen',['Yes','No'])
Ips=st.selectbox('IPS',['Yes','No'])
Screen_size=st.number_input('Screen Size')
Resolution=st.selectbox('Screen Resolution',['1920x1080','1366x738','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1400','2304x1440'])
CPU=st.selectbox('CPU_Brand',df['CPU'].unique())
HDD=st.selectbox('HDD in GB',[0,128,256,512,1024,2048])
SDD=st.selectbox('SDD in GB',[0,8,128,256,512,1024])
GPU=st.selectbox('GPU_Brand',df['GPU'].unique())
Op_Sys=st.selectbox('OS',df['Op_Sys'].unique())

if st.button('Predict Price'):
    ppi=None
    if Touchscreen=='Yes':
        Touchscreen=1
    else:
        Touchscreen=0
    
    if Ips=='Yes':
        Ips=1
    else:
        Ips=0

    X_res=int(Resolution.split('x')[0])
    Y_res=int(Resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/Screen_size


    query=np.array([Company,TypeName,Ram,weight,Touchscreen,Ips,ppi,CPU,HDD,SDD,GPU,Op_Sys])
    query=query.reshape(1,12)
    st.title("The Predicted Price is : "+str(int(np.exp(pipe.predict(query)[0]))))
