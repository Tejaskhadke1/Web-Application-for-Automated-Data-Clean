import streamlit as st
import plotly_express as px
import pandas as pd
from streamlit_option_menu import option_menu
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import re
import time
import itertools
from ngram import NGram
from dateutil.parser import parse

from chart_studio import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)

# Extra options
pd.options.display.max_rows = 30
pd.options.display.max_columns = 25

# Show all code cells outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

import os
from IPython.display import Image, display, HTML

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from scipy.spatial.distance import pdist, squareform
from scipy import stats

import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Confirigation
st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("DATA VISUALIZATION APP")

# Add a Sidebar
st.sidebar.subheader("UPLOAD YOUR FILE HERE")


# Add a option
with st.sidebar:
    selected = option_menu("HOW MANY FILES YOU WISH TO UPLOAD", ["One", "Two"], 
        icons=['one', 'two'], menu_icon="cast", default_index=1)

        
global df_1
global df_2
global df
if selected == "One":
    uploaded_file = st.sidebar.file_uploader(label="UPLOAD YOUR CSV OR EXCEL FILE.",type=['csv','xlsx'])
 
    if uploaded_file is not None:
        print(uploaded_file)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
        
    try:
        st.write(df)
        df["id"] = df.index
        
    except Exception as e:
        
        st.write("Please Upload file to the Application")

else:

    global option1
    global option2


    
    def merge_files(df_1, df_2):
        return pd.merge(
        df_2, 
        df_1, 
        how='outer',
        left_on = option1,
        right_on = option2)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')



    # uploads
    file_one = st.sidebar.file_uploader('Please Upload File One')
    file_two = st.sidebar.file_uploader('Please Upload File Two')

# check if files are uploaded

    if file_one is not None:
        
        try:
            df_1 = pd.read_csv(file_one)
        except Exception as e:
            df_1 = pd.read_excel(file_one)

            
    if file_two is not None:

        try:
            df_2 = pd.read_csv(file_two)
        except Exception as e:
            df_2 = pd.read_excel(file_two)


     

    if file_one  is not None and file_two is not None:
        option1 = st.sidebar.selectbox('PLEASE PICK FIELD FROM FILE_ONE',set(list(df_1.columns)))
        option2 = st.sidebar.selectbox('PLEASE PICK FIELD FROM FILE_TWO',set(list(df_2.columns)))

    if st.sidebar.button('MERGE'):
        if (file_one is not None) & (file_two is not None):
            df = merge_files(df_1, df_2)
            st.write(df)
            df["id"] = df.index
            
    elif (file_one == None) & (file_two is not None):
        st.error('Please upload file_one')
    elif (file_one is not None) & (file_two == None):
        st.error('Please upload file_two')
    else:
        st.write('Please upload your files')



from IPython.display import display


# NUMBER 1



def duplicate_clean (col):
    if (not df[col].is_unique):
        time.sleep(0.05)
        df.drop_duplicates([col], keep='first', inplace=True)
        return
      
#NUMBER 2

def clean_non_numerical (col):
    
    unique_vals = df[col].unique()
    if (len(unique_vals) > 30):
        return duplicate_clean(col)
    
    def is_date(string):
        try: 
            parse(string)
            return True
        except ValueError:
            return False
    
    for i, row_value in df[col].head(5).iteritems():
        if is_date(df[col][i]):
            return
    
    if ('country' in col) or ('COUNTRY' in col) or ('Country' in col):
        invalid_df_us = df[(df[col] == 'United States') | (df[col] == 'United States of America') | (df[col] == 'USA') | (df[col] == 'usa') | (df[col] == 'us')]
        if invalid_df_us.shape[0] > 0:
    
            df.loc[(df[col] == 'United States') | (df[col] == 'United States of America') | (df[col] == 'USA') | (df[col] == 'usa') | (df[col] == 'us'), [col]] = 'US'
        invalid_df_uk = df[(df[col] == 'United Kingdom') | (df[col] == 'uk')]
        if invalid_df_us.shape[0] > 0:
           
            df.loc[(df[col] == 'United Kingdom') | (df[col] == 'uk'), [col]] = 'UK'
        

    unique_vals = ['None' if x is np.nan else x for x in unique_vals]
    unique_vals = ['None' if v is None else v for v in unique_vals]

    
    df[col] = df[col].replace(r"^ +| +$", r"", regex=True)
    
    
    def title_case(string):
        if pd.isnull(string):
            return None
        return re.sub(r"[A-Za-z]+('[A-Za-z]+)?", lambda word: word.group(0).capitalize(), string)
    df[col] = df[col].apply(lambda x: title_case(x))
    
    
    def ngram_compare(a,b):
        if NGram.compare(a,b) > 0.4:
           
            time.sleep(0.05)
            combine = 'y'      #idhar game lag sakta hai
            if combine == 'yes' or combine == 'y':
                while True:
                    time.sleep(0.05)
                    combine_to = a
                    if combine_to == a:
                        df[col] = df[col].replace({b: a}, regex=True)
                        return
                    elif combine_to == b:
                        df[col] = df[col].replace({a: b}, regex=True)
                        return
                    else:
                        continue
    unique_vals = df[col].unique()
    for pair in itertools.combinations(unique_vals, r=2):
        ngram_compare(*pair)
    print("\n")
    
    unique_vals = df[col].unique()
    unique_vals = ['None' if v is None else v for v in unique_vals]



#NUMBER 3

    
def clean_numerical (col, df):
    time.sleep(0.05)
    print("\n")
    
    
    duplicate_clean(col)
    print("\n")
    
    
    

    if ('zip' in col) or ('ZIP' in col):
        invalid_df = df[(df[col] < 1) | (df[col] > 99950)]
        if invalid_df.shape[0] > 0:
            
            
            df.loc[(df[col] < 1) | (df[col] > 99950), [col]] = np.nan

    
    
    if ('age' in col) or ('AGE' in col):
        invalid_df = df[(df[col] < 0) | (df[col] > 120)]
        if invalid_df.shape[0] > 0:
            
            df.loc[(df[col] < 0) | (df[col] > 120), [col]] = np.nan
            
        
def main_1(df):
    import warnings
    warnings.filterwarnings('ignore')
    
    for column in df:
        if (df[column].dtype == object):
            clean_non_numerical (column)
            

    print("NON_NUMERICAL_DATA_CLEANED")

    
def main_2 (df):
    import warnings
    warnings.filterwarnings('ignore')
    
    for column in df:
        if(df[column].dtype == 'int64' or df[column].dtype == 'float64'):
            clean_numerical (column, df)

    smart_dup = pd.DataFrame(1 - squareform(pdist(df.set_index('id'), lambda u,v: (u != v).mean())))
    smart_dup.values[[np.arange(smart_dup.shape[0])]*2] = 0
    smart_dup = smart_dup.mask(np.triu(np.ones(smart_dup.shape, dtype=np.bool_)))
    duplicates = smart_dup[smart_dup > 0.8]
    dup_list = np.stack(duplicates.notnull().values.nonzero()).T.tolist()
        
    for dup in dup_list:
        display(df.iloc[[dup[0],dup[1]],:])
    for dup in dup_list:
        try:
            df.drop(df.index[dup[0]], inplace=True)
        except:
            continue
    print("** Numerical Data Cleaning Complete! **")
    
    
    
def main_3(df):
    for column in df:
        
        if df[pd.isnull(df[column])].shape[0] > 0:
            
            
            
            
            if (df[column].dtype == 'int64' or df[column].dtype == 'float64'):
                del_option = "5"
            else:
                del_option = "1"
                
            
            if del_option == "1" or del_option == "(1)":
                continue
                
            elif del_option == "2" or del_option == "(2)":
                df.dropna(subset=[column],how='any',inplace=True)
                
            elif del_option == "3" or del_option == "(3)":
                
                df.drop(columns=[column], inplace=True)
            elif del_option == "5" or del_option == "(5)":
               
                time.sleep(0.05)
                imp_option = 'a'
                if imp_option == 'a' or imp_option == '(a)':
                    knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
                    df[column] = knn_imputer.fit_transform(df[[column]])
                elif imp_option == 'b' or imp_option == '(b)':
                    mice_imputer = IterativeImputer()
                    df[column] = mice_imputer.fit_transform(df[[column]])
        print("\n")
    print("** Missing Data Cleaning Complete! **") 


def main_4 (df):
    pr3 = ProfileReport(df, explorative=True)
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr3)

def main ():
    main_1(df)

    main_2(df)

    main_3(df)
    
    main_4(df)
    
    print(df)

try:
    if __name__ == main():
        main()

except Exception as e:
    print(e)
    pass



