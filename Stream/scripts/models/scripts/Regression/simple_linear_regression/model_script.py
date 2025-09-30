import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from constants import DataManager
data_manager = DataManager()

def model_script(df,features,target,edit):
                                                        # Prepare data
                    X = df[features].values
                    y = df[target].values
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Store results
                    model_results = {
                        'model': model,
                        'metrics': {
                            'MSE': mse,
                            'R2 Score': r2,
                            'Coefficients': model.coef_,
                            'Intercept': model.intercept_
                        },
                        'features': features,
                        'target': target
                    }
                    param_list =  [
                                {
                                "name":"features",
                                "value":features
                                },
                            {
                                "name" : "target",
                                "value":target
                            }
                            ]
                    # st.session_state.pipeline.setdefault("ML", [])
                    LR_model=   DataManager.create_Linear_REG_Model(
                                "LRM",
                                param_list,
                                st.session_state.selected_trans
                        ) 
                    # st.write(LR_model)
                    # st.session_state.pipeline['ML'].append( 
                    #     LR_model                             
                    # )
                    # Check if an 'LRM' item exists in the list
                    # lrm_exists = any(item.get('name') == 'LRM' for item in st.session_state.pipeline['ML'])

                    if edit:
                        # Replace the existing LRM entry
                        st.session_state.pipeline['ML'] = [
                            item if item.get('name') != 'LRM' else LR_model
                            for item in st.session_state.pipeline['ML']
                        ]
                    else:
                        # Append if no LRM exists
                        st.session_state.pipeline['ML'].append(LR_model)
                    st.success("Model created successfully!")
                    return model_results
def validate_model(params):
        if len(params['features']) == 0:
            st.error("Please select at least one feature column")
            return False
        elif params['target'] in params['features']:
            st.error("Target column cannot be one of the features")
            return False
        return True