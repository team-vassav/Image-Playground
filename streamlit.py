
import streamlit as st
import pandas as pd

st.write(""" # Welcome to VASSAV App ! # """)

def data():
    DATA = 'CDAC.csv'
    df = pd.read_csv(DATA)


    df = pd.DataFrame(df, columns=["Employee", "Basic/Consolidated Pay"])

    # plot the dataframe
    st.line_chart(df)

agree = st.checkbox('I agree to the terms and conditions')

if agree:
     st.write('Great!')
     if st.button('Next'):
         '''CDAC Salary'''
         data()
