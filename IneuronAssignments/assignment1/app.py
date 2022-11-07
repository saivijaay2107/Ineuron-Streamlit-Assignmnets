import pandas as pd
import numpy as np
import streamlit as st
import time
import scipy as sp

st.title("Air pollution calculation")
# Formula Ip = [IHi – ILo / BPHi – BPLo] (Cp – BPLo) + ILo
def Air(IHi,ILo,BPHi,BPLo,Cp):
    AQI =  ((IHi - ILo) / (BPHi - BPLo))*(Cp - BPLo) + ILo
    return AQI    


IHi=st.number_input("Enter AQI value corresponding to BPHi")
ILo =st.number_input("Enter AQI value corresponding to BPLo")
BPHi=st.number_input("Enter concentration breakpoint greater than or equal to Cp",min_value=0.01)
BPLo=st.number_input("Enter concentration breakpoint less than or equal to Cp")
Cp=st.number_input("Enter truncated concentration of pollutant")

agree = st.button("Predict Air Quality")
if agree:
    with st.spinner():
        time.sleep(3)
        st.write("The air quality value is",Air(IHi,ILo,BPHi,BPLo,Cp))


