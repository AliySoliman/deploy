import streamlit as st    
from constants import DataManager
import transform ,editor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from models import models_execution
data_manager = DataManager()
import machine
from models import html
from models.scripts.Regression.simple_linear_regression.model_script import model_script
from models.scripts.Regression.simple_linear_regression.model_report import model_report
from models import  model_components
def show_model_page(edit,model_data,model_name,data_uploaded,reset):
            """Detailed model implementation page"""
            title = f"{model_data['model name'] if edit else model_name} Implementation"
            st.markdown(data_manager.label_style, unsafe_allow_html=True)
            st.markdown("""
            <div class="custom-container">
                <h2 class="custom-header">🚀"""+title+"""</h2>
            </div>
            """, unsafe_allow_html=True)
            if reset:
                # if 'selected_data' in st.session_state:
                #     del st.session_state['selected_data']
                # if 'selected_trans' in st.session_state:
                #      del st.session_state['selected_trans']
                if 'model_data' in st.session_state:
                    del st.session_state['model_data']
                if 'comments' in st.session_state:
                    del st.session_state['comments']
            # INITIALIZATION
     
 
            # if 'selected_trans' in st.session_state:
            #     del st.session_state['selected_trans']
                # st.write(st.session_state.selected_data)
            if 'selected_trans' not in st.session_state :
                st.session_state.selected_trans = []
            if data_uploaded:
                st.write(edit)
                if edit :
                    
                    # st.session_state.comments=model_data['comments']
                    if 'selected_trans' not in st.session_state :
                        st.session_state.selected_trans = model_data['transformations']
                    if 'selected_data' not in st.session_state  :
                        st.session_state.selected_data = transform.apply_selected_transformations(model_data['transformations'])
                    default_trans =  model_data['transformations']
                else :
                     if 'selected_data' not in st.session_state :
                        st.session_state.selected_data = st.session_state.df_original
                    # else : st.session_state.selected_trans = []
                     default_trans =  []
            else : 
                st.session_state.selected_data = pd.DataFrame()
                # st.session_state.selected_trans = []
                default_trans = []
            if 'show_data_transformations' not in st.session_state:
                st.session_state.show_data_transformations = False
            # if st.session_state.current_page == 'main':
            #    print('main')
            #    del  st.session_state['model_data']
            #    del st.session_state['selected_data']
            # Back Button
            if st.button("Back to Main"):
                st.session_state.current_page = "main"
                if 'selected_data' in st.session_state:
                    del st.session_state['selected_data']
                if 'selected_trans' in st.session_state:
                     del st.session_state['selected_trans']
                st.rerun()
# ####################################################################
            #  MODEL DESCRIPTION
            st.markdown(models_execution.get_model_data(model_name,"model_description") , unsafe_allow_html=True)
            #  MODEL REFERENCE CODE 
            with st.expander("reference code"):
                st.markdown(
                    models_execution.get_model_data(model_name,"model_reference") , unsafe_allow_html=True)
######################################################################
            # Data transformations dialog
            if st.button("Choose Your Data Transformations"):
                st.session_state.show_data_transformations = not st.session_state.show_data_transformations
                st.write(st.session_state.show_data_transformations,"show")

            if st.session_state.show_data_transformations:
            
                with st.expander("Data Transformation Options", expanded=True):
                    trans =st.session_state.pipeline['transformations']
                    selected_trans = st.multiselect(
                    "Select columns to include:",
                    options=[step["name"] for step in trans]if data_uploaded else [],
                    default= default_trans,
                    key="data_transform_columns" 
                    
                    ) 
                    # st.write(st.session_state.selected_data)
                    if not st.session_state.selected_data.empty:
                        st.dataframe(transform.apply_selected_transformations(selected_trans))
                    else :
                        # st.write("else")
                        st.error('Please Load you data first', icon="🚨")
       

                    if st.button("Apply Transformations", key="apply_transformations"):
                        st.write(selected_trans)
                        st.session_state.selected_trans = selected_trans 
                        st.session_state.selected_data = transform.apply_selected_transformations( st.session_state.selected_trans)
                        st.success("Data transformations applied successfully!")
                        # st.session_state.show_data_transformations = False
                        st.rerun()
                        
            
            try:
            # Model configuration section
                st.markdown(data_manager.label_style, unsafe_allow_html=True)
                st.markdown("""
                <div class="custom-container">
                    <h2 class="custom-header">🚀 Model configuration</h2>
                </div>
                """, unsafe_allow_html=True)
                config_params ={'model_data':model_data,"edit":edit}
                validate_params = {"params":models_execution.execute_model(model_name,"config",config_params)}
                script_params=validate_params["params"]
                # st.write(script_params)
                col3,col4 = st.columns([1,1])
                with col3:
                    if st.button("Create Model"if not edit else "Update Model",):
                        if models_execution.execute_model(model_name,"validate",validate_params)  :
                            ## Model Execution 
                            # params = {"df":st.session_state.selected_data,"features":features,"target":target,"edit":edit}
                            st.session_state.model_results = models_execution.execute_model(model_name,"script",script_params)
                                

                if edit: 
                    with col4:
                        if st.button("Delete Model") :
                            machine.delete_model(model_name=model_data['model name'])
                            st.success('Model deleted')
                    
                # Show report if model was created
                if 'model_data' not in st.session_state and edit:
                    st.session_state.model_data = model_data
                    # st.write(st.session_state.model_data)
                    ## Model Report 
                    ## you will make a function named_model_report and call it here 
                    ## The function will be in the molder folder like i did 
                    ## Execute model will take the model name and the type of the execution wheter report or script
                if 'model_results' in st.session_state:
                    if st.button("Show Report", key="show_report_button"):
                        models_execution.execute_model(model_name,"report",{})
                        # model_report()
            ## Here is the comments section 
            ## same with every model
                model_components.comments_section(edit)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()