# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import time
import pickle
import streamlit  as st
from streamlit_lottie import st_lottie
import streamlit_option_menu
import requests
import json
from PIL import Image
from streamlit_extras.dataframe_explorer import dataframe_explorer

# SETTING PAGE CONFIGURATION
st.set_page_config(page_title='Churn Analysis',layout='wide')

# SETTING STREAMLIT STYLE
streamlit_style = """
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
                        
                        html,body,[class*='css']{
                            font-family:'sans-serif';
                        }
                        </style>
                  """
st.markdown(streamlit_style,unsafe_allow_html=True)

# LOADING LOTTIE FILES
def load_lottier(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# LOADING DATASET  
df = pd.read_csv('Telco-Customer-Churn.csv')
df.index = range(1,len(df)+1)

def main():
    # USING LOCAL CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')
    
    # CREATING NAVIGATION BAR WITH OPTION MENU
    selected = streamlit_option_menu.option_menu(menu_title=None,
                                                 options=['Data','Analysis'],
                                                 icons=['activity','graph-down'],
                                                 menu_icon='list',default_index=0,
                                                 orientation='horizontal',
                                                 styles={
                                                     'container': {'padding':'0!important','backgorund-color':'#white'},
                                                     'icon':
                                                         {'color':'yellow','fontsize':'25px'},
                                                     'nav-link':
                                                         {'fontsize':'25px','text-align':'middle','margin':'0px','--hover-color':'grey'},
                                                      'nav-link-selected':{'background-color':'blue'}  
                                                 })
    
    def title(string):
        st.markdown(f"<h1 style='color:#00FFFF';font-size:40px>{string}</h1>",unsafe_allow_html=True)
    def header(string):
        st.markdown(f"<h2 style='color:#FF00FF';font-size:40px>{string}</h2>",unsafe_allow_html=True)
    def subheader(string):
        st.markdown(f"<h3 style='color:#FCB325';font-size:40px>{string}</h3>",unsafe_allow_html=True)
    def plot_subheader(string):
        st.markdown(f"<h3 style='color:#41FB3A';font-size:40px>{string}</h3>",unsafe_allow_html=True) 
    def inference_subheader(string):
        st.markdown(f"<h3 style='color:#FCB325';font-size:40px>{string}</h3>",unsafe_allow_html=True) 
               
    
    # CREATING HOME PAGE
    if selected=='Data':
        title('Churn Analysis')
        #lottie_coding1 = load_lottier('https://assets4.lottiefiles.com/packages/lf20_jcTRQijNzu.json')
        # CONTAINER TO DESCRIBE ABOUT CHURN ANALYSIS IN GENERAL
        with st.container():
            text_column,image_column = st.columns((2,1))
            with text_column:
                subheader('What is Churn Analysis?')
                st.write("""Customer churn analysis is the process of using the churn data to understand behavior of customers in a business.We can know some of the reasons why a customer is leaving.For better growth in a business,business owners should understand the major characteristics or facilities that they can provide to their customers.""")
                st.write("""Customer Churn is one of the most problems for businesses such as Credit Card companies, cable service providers, SASS and telecommunication companies worldwide. Even though it is not the most fun to look at, customer churn metrics can help businesses improve customer retention.""")
                st.write("""Churn analysis is the evaluation of a company’s customer loss rate in order to reduce it. Also referred to as **Customer Attrition Rate**, churn can be minimized by assessing a business product and how people use it""")
            with image_column:
                #st_lottie(lottie_coding1,height=300,key='lottie_coding1') 
                st.image(Image.open('churn_customers_image.png'),width=500)
        st.write('- - -')       
 
        # DESCRIBING CHURN ANALYSIS IN TELECOM INDUSTRY        
        subheader('Churn Analysis in Telecom Industry')
        st.write('We have Churn data of a Telecom Industry,dataset and feature description of features are displayed below.I did some analysis with better visualizations to understand the data.')
        header('Dataset')
        st.dataframe(dataframe_explorer(df),use_container_width=True)
        
        st.write('- - -')
        
        # FEATURES OF DATASET
        header('Features of Dataset')
        description = ['Unique Customer ID',
              'Gender of Customer',
              'describes whether Customer is a SeniorCitizen or not(0 means N0, 1 means Yes)',
              'describes whether Customer is a Partner of company or not(Yes/No)',
              'describes whether Customer is a Dependent or not(Yes/No)',
              'Number of months since customer stayed with company',
              'describes whether Customer is provided with Phone Service or not(Yes/No)',
              'describes whether Customer has Multiple Lines or not(only if they are provided with Phone Service)',
              'describes the type of Internet Service provided for Customer',
              'describes whether Customer is provided with Online Security or not(Yes/N0)',
              'describes whether Customer is provided with Online Backup or not(Yes/N0)',
              'describes whether Customer is provided with Device Protection or not(Yes/N0)',
              'describes whether Customer is provided with Tech Support or not(Yes/N0)',
              'describes whether Customer is provided with Streaming TV or not(Yes/N0)',
              'describes whether Customer is provided with Streaming Movies or not(Yes/N0)',
              'describes type of Contract between Customer and Company(Month-Month or One-Year or Two-Years)',
              'describes whether Customer is provided with Paperless Billing or not(Yes/N0)',
              'describes Payment Method(Electronic check,Mailed check,Automatic Bank transfer,Credit card)',
              'Monthly Charges of Customer',
              'Total Charges of Customer',
              'describes whether Customer was Churn Customer or not(Target Feature)']        
        feature_description_df = pd.DataFrame({'Feature Name':df.columns.tolist(),
             'Description':description},index=range(1,22))
        st.table(feature_description_df)
        st.write('- - - ')
        with st.container():
            pie_chart_column,text_column = st.columns((2,1))
            with pie_chart_column:
                plot_subheader('Churn Percentage')
                trace = go.Pie(labels=df['Churn'].value_counts().keys().tolist(),
                values = df["Churn"].value_counts().values.tolist(),
                marker = dict(colors = ['royalblue','red'],
                          line = dict(color = "black", width = 1.0)
                          ),
                rotation = 90,
                hoverinfo = "label+value+text",
                hole = .5,pull=[0,0.05])
                layout = go.Layout(dict(plot_bgcolor = "rgb(0,0,0)",
                        paper_bgcolor = "rgb(0,0,0)",
                       width=800,height=500)
                  )
                layout = layout
                fig = go.Figure(data=[trace],layout=layout)
                fig.update_traces(textfont_size=20)
                fig.update_layout(legend=dict(font=dict(size=20)))
                fig.update_layout(hoverlabel=dict(bgcolor='black',
                                              font_size=16,
                                              font_family='Rockwell',
                                              font_color='white'))
                st.plotly_chart(fig,use_container_width=True)
            with text_column:
                st.write(' ');st.write(' ');st.write(' ');st.write('  ');st.write(' ');st.write(' ');st.write(' ');st.write('')
                st.write('In dataset,')
                st.markdown('- **Percentage of Churn Customers is 26.6%**')  
                st.markdown('- **Percentage of Non-Churn Customers is 26.6%**')  
        st.write('- - -')    
        
        with st.container():
            col_1,col_2,col_3 = st.columns((1,1,1))
            with col_2:
                plot_subheader('Correlation of Target Column')
        
        # CONTAINER TO DISPLAY CORRELATION PLOT        
        with st.container():
            
            # DATA PREPARATION
            df['Partner'] = df['Partner'].replace(to_replace=['Yes','No'],value=[1,0])
            df['Dependents'] = df['Dependents'].replace(to_replace=['Yes','No'],value=[1,0])
            df['PhoneService'] = df['PhoneService'].replace(to_replace=['Yes','No'],value=[1,0])
            df['PaperlessBilling'] = df['PaperlessBilling'].replace(to_replace=['Yes','No'],value=[1,0]) 
               
            # CREATING DUMMIES DATA FRAME
            dummies_df = pd.get_dummies(df.drop('customerID',axis=1),drop_first=False)
            dummies_corr_df1 = dummies_df.corr()
            dummies_corr_df1.drop('Churn_No',axis=1,inplace=True)
            dummies_corr_df1.drop(['Churn_Yes','Churn_No'],axis=0,inplace=True)
            Churn_Yes_df =  dummies_corr_df1['Churn_Yes'].sort_values()
            
            # PLOTTING CORRELATION FIGURE
            fig = px.bar(data_frame=Churn_Yes_df,height=800,width=1200,labels={'index':'Feature','value':'Correlation Value'},color_discrete_sequence=['#ff6600'])
            fig.update_layout(hoverlabel=dict(bgcolor='black',
                                              font_size=16,
                                              font_family='Rockwell',
                                              font_color='white'))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
            
            # DESCRIPTION ABOUT CORRELATION PLOT
            inference_subheader('Inference:')
            st.write('Target having high positive correlation with a feature means on increasing that feature value,target is also increasing.High negative correlation with a feature means on increasing that feature value,target is decreasing.')
            st.write('Above barplot shows the dependency of Churn Rate with other Features.In that plot,highly positively correlated fetaures are reason for higher Churn Rate.And highly negative correlated features are reason for lower Churn Rate.')
            st.markdown('**Churn Rate is highly increasing,when**')
            st.markdown('- Customer is not provided with Online Security,Tech Support.')
            st.markdown('- Customer has only One Month Contract with company.')
            st.markdown('**Churn Rate is highly decreasing,when**')
            st.markdown("- Customer's tenure length is high,that means more months of realtionship with company.")
            st.markdown('- Customer having a Contract of Two Years with company,is less likely to Churn')
            st.markdown('**Churn Rate has less dependency**,with providing of Streaming Movies and providing of MultiLines Phone Service to Customer.')
            
    if selected == 'Analysis':
        with st.container():
            header('Cohort Analysis')        
            st.write('''Cohort Analysis is a form of behavioral analytics that takes data from a given subset, such as a SaaS business, game, or e-commerce platform, and groups it into related groups rather than looking at the data as one unit. The groupings are referred to as cohorts. They share similar characteristics such as time and size.''')
            st.write('''Cohort analysis technique is typically used to make it easier for organizations to isolate, analyze, and detect patterns in the lifecycle of a user, to optimize customer retention, and to better understand user behavior in a particular cohort.''')
        st.write('- - -')
        with st.container():
            plot_subheader('Churn Rate and length of Tenure')
        with st.container():
            churn_rate_plot,text_plot = st.columns((3,1))
            with churn_rate_plot:
                # CREATING NO_CHURN AND YES_CHURN DATA   
                no_churn = df.groupby(['Churn','tenure']).size()['No']
                yes_churn = df.groupby(['Churn','tenure']).size()['Yes']
                # GETTING CHURN RATE 
                churn_rate =  100 * yes_churn / (yes_churn + no_churn)
                # PLOTTING LINE PLOT OF CHURN RATE WITH TENURE COLUMN
                fig = px.line(x=churn_rate.index,y=churn_rate.values,labels={'x':'Tenure','y':'Chrun Rate'},height=600,width=900,template='simple_white')
                fig.update_layout(hovermode='x')
                fig.update_traces(line_color='#FC4017')
                fig.update_layout(hoverlabel=dict(bgcolor='black',
                                              font_size=16,
                                              font_family='Rockwell',
                                              font_color='white'))
                st.plotly_chart(fig,use_container_width=True)
            with text_plot:
                st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ');st.write(' ')    
                st.write('As seen from the Line plot,')
                st.markdown('- Churn Rate(Attrition rate) is decreasing while tenure(Number of months customer engaged with company) is increasing.')
        st.write('- - -')
        # CONTAINER FOR BROADER COHORT GROUPS        
        with st.container():
            plot_subheader('Broader Cohort Groups')
            st.write('''Time-based cohorts are customers who signed up for a product or service during a particular time frame. Analyzing these cohorts shows the customers’ behavior depending on the time they started using a company’s products or services.''')
            st.write('As tenure is number of months a customer being engaged with company.I clustered customers based on tenure length.Below Scatter plot is in between the features Monthly Charges and Total Charges,coloring with Tenure Cohorts.')
    
            # CREATING A FUNCTION THAT CREATES COHORTS BASED ON TENURE(NUMBER OF MONTHS)
            def create_group(tenure_value):
                if tenure_value in range(1,12+1):
                    return '0-12 Months'
                elif tenure_value in range(13,24+1):
                    return '12-24 Months'
                elif tenure_value in range(25,48+1):
                    return '24-48 Months'
                else:
                    return 'over 48 Months'
            df['Tenure Cohort'] = df['tenure'].apply(func=create_group)
            fig = px.scatter(data_frame=df,x='MonthlyCharges',y='TotalCharges',color='Tenure Cohort',width=1200,height=700)
            fig.update_layout(hoverlabel=dict(bgcolor='black',
                                              font_size=16,
                                              font_family='Rockwell',
                                              font_color='white'))
            fig.update_layout(legend=dict(font=dict(size=15)))
            fig.update_traces(textfont_size=20)
            st.plotly_chart(fig,use_container_width=True) 
            
        st.write('- - -')
        # PREDICTIVE MODELLING
        with st.container():
            header('Predictive Modelling')
            st.write('By using the dataset,i created a Machine Learning Classification model.I did follow the below steps.')
            st.markdown('- Data Exploration and Visualizations with Pandas,Matplotlib,Seaborn,Plotly Libraries of Python.')
            st.markdown('- Making Dummy Variables for Categorical features.')
            st.markdown('- Getting Correlation/Dependency of Target with Features of dataset.')
            st.markdown('- Cohort Analysis by making Cohort groups based on Tenure Column.')
            st.markdown('- Dealing with Imbalanced data.')
            st.markdown('- Splitting the data into X-Features and y-Target.')
            st.markdown('- Creating models for Machine Learning Classification algorithms and performing **Hyperparameter tuning**.')
            st.markdown('- Comparing the performance of models with **Confusion Matrix** and **Accuracy Score**')
            st.write(' ')
            st.write('Machine Learning Algorithms used are')
            space_col,list_column,space_col = st.columns((1,1,1))
            with list_column:
                st.markdown('- Logistic Regression')
                st.markdown('- Support Vector Machine Classifier')
                st.markdown('- K-Nearest Neighbours Classifier')
                st.markdown('- Decision Tree Classifier')
                st.markdown('- Random Forest Classifier')
                st.markdown('- Ada-Boost Classifier')
                st.markdown('- Gradient Descent Classifier')
                st.markdown('- Extreme Gradient Descent Boosting (**XGBoost**) Classifier')
            with st.container():
                subheader('Accuracy Scores')
                st.write('**Accuracy Scores of Classification models on Test data**') 
                accuracy_df = pd.read_csv('Final_Accuarcy_df.csv')
                accuracy_df.index = accuracy_df.index + 1
                with st.container():
                    df_column,space_column = st.columns((2,1))
                    with df_column:
                        st.table(accuracy_df)  
                        
                st.write('As per above table,')
                st.markdown('- Ada-Boost model is performing better with our data.')
                st.markdown('- We can observe Accuracy scores of Boosting algorithms Ada-Boost,Gradient Boost,XG-Boost are same.')
                st.write(' ')
                st.write('Python code for Machine Learning [Here](https://github.com/TRGanesh/Telecom-Churn-Analysis/blob/main/Customer_Churn_Classification.ipynb)')
                st.write('Python code for Streamlit Web Page [Here](https://github.com/TRGanesh/Telecom-Churn-Analysis/edit/main/churn_analysis_app.py)')
       
                        
                

            
            
if __name__ == '__main__':
    main()        
