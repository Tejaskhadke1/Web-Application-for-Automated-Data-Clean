import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



# Web App Title
st.markdown('''
# **AUTO VISUALIZATION APP**
This is the **EDA App** created in **Streamlit** using the **pandas-profiling** library.

''')

# Upload CSV data
with st.sidebar.header('1. Upload your files here'):
    uploaded_file = st.sidebar.file_uploader("Upload file",type=["csv","xlsx","xls"])
    merge_file = st.sidebar.file_uploader("Upload another file",type=["csv","xlsx","xls"])
    st.write(uploaded_file)
    st.write(merge_file) 
# Pandas Profiling Report

    @st.cache
    def load_csv(file):
        csv = pd.read_csv(file)
        return csv
    @st.cache
    def load_excel(file):
        excel = pd.read_excel(file)
        return excel

if uploaded_file is not None:
    try:
        df1 = pd.read_csv(uploaded_file)
    except Exception as e:
        df1 = pd.read_excel(uploaded_file)

    
    st.header('**Input DataFrame**')
    st.write(df1)
    st.write('---')
    
        
    if merge_file is not None:
        try:
            df2 = pd.read_csv(merge_file)
        except Exception as e:
            df2 = pd.read_excel(merge_file)


        st.header('**Input DataFrame**')
        st.write(df2)
        st.write('---')


        option1 = st.selectbox('Which column to connect from Table 1 ?',set(list(df1.columns)))
        option2 = st.selectbox('Which column to connect from Table 2 ?',set(list(df2.columns)))
        option3 = st.selectbox('Which join to perform',('left', 'right', 'inner','outer'))
        if st.button("Run EDA"):
            if option3 == 'left' or option3 == 'right':
                data = pd.merge(df1,df2,how = option3,left_on = option1,right_on = option2)
                st.write(data)
                pr = ProfileReport(data, explorative=True)
                st.header('**Pandas Profiling Report**')
                st_profile_report(pr)
            else:
                data = pd.merge(df1,df2,how = option3)
                st.write(data)
                pr = ProfileReport(data, explorative=True)
                st.header('**Pandas Profiling Report**')
                st_profile_report(pr)
    else:
        if st.button("Run EDA"):
            st.write(df1)
            pr = ProfileReport(df1, explorative=True)
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

else:
    st.info('Awaiting for file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
