import streamlit as st 
import mysql.connector
import pandas as pd
from streamlit_option_menu import option_menu
import base64
import plotly_express as px 
import matplotlib.pyplot as plt

st.set_page_config(page_title="Admin Panel",layout="centered",page_icon=":blue_car:")


st.markdown("""
<style>
.css-9s5bis.edgvbvh3{
    visibility :hidden;
}
.css-1q1n0ol.egzxvld0{
    visibility : hidden;
}
</style>
""",unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('admin.png') 
with st.sidebar:
 selected= option_menu(
        menu_title="Main Menu",
        options=["Admin","Visitor Detail"],
        default_index=0,
        icons=["person-fill","people-fill"],
        menu_icon="cast",
)
 st.markdown(
    """
    <style>
   
    .footer {
        margin-top: 370px;
        padding: 10px 0;
        text-align: center;
        font-size: 16px;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div class="footer">
        &copy; 2024 Anukul Srivastava
    </div>
    """,
    unsafe_allow_html=True
)
####################################################################
#database Work
####################################################################

conn = mysql.connector.connect(
            host="localhost",
            user="sqluser",
            password="password",
            database="python_db")
cursor = conn.cursor()              
cursor.execute("Select id,Name,Email,T_Date from user_table")
show_result = cursor.fetchall()
df = pd.DataFrame(show_result,columns=['Id','Name','Email','T_Date'])
cursor.execute('Select Count(T_Date) from user_table')
a = cursor.fetchone()
b=a[0]



######################################################
#'''Admin'''
#####################################################
if selected == "Admin":
    st.markdown("<h1 style='text-align:center;color:white;'>Welcome to Dashboard</h1>",unsafe_allow_html=True)
    st.markdown("----")
                                                                                                                                                                        
    st.markdown(f"<h3 style = 'color:white'>Visitor Count:{b} </h3>",unsafe_allow_html=True)



if selected=='Visitor Detail':
    st.markdown("<h1 style ='text-align:center;color:white'>DashBoard</h1>",unsafe_allow_html=True)
    # Check = ['Visitor Detail','Plot Graph']
    Check = ['Visitor Detail']

    st.write("<h4 style ='color:white'>Select a Option</h4>",unsafe_allow_html=True)
    Options = st.selectbox('',Check)
    if Options == 'Visitor Detail':
        df
    # elif Options == 'Plot Graph':
    #     p= df.groupby('T_Date')['Id'].count()
    #     s = st.dataframe(p)
    #     st.markdown(s)


