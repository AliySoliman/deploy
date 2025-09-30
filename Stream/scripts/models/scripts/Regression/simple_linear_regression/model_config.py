import streamlit as st
def model_config(model_data,edit):        
        numeric_cols = [col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
        # Feature and target selection
        col1, col2 = st.columns(2)
        with col1:
            features = st.multiselect(
                "Select feature columns:",
                options=[col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] if len(numeric_cols) != 0 else [] ,
                    default=model_data['model param'][0]['value']if edit else []
                )
        
        with col2:
            if len(numeric_cols) == 0:
                st.warning("No numeric columns available for target selection")
                target = None
            else:
                # Safe index calculation for selectbox
                default_index =numeric_cols.index(
                        model_data['model param'][1]['value']
                        )if edit else 0
                target = st.selectbox(
                    "Select target column:",
                    options=numeric_cols,
                    index=default_index if edit else len(numeric_cols)-1,  # Ensure index is valid
                    key="target_column"
                )    
                return {"features":features,"target":target,"df":st.session_state.selected_data,"edit":edit}
SLR_model_reference_code = """
                <div class="code-container">
                    <div class="code-header">
                        <span>MODEL DESCRIPTION</span>
                        <span>Linear Regression</span>
                    </div>
                    <div>
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
                            st.session_state.model_results = {
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
                            </div>
                </div>
                """
SLR_model_description = """
            <div class="code-container">
                <div class="code-header">
                    <span>MODEL DESCRIPTION</span>
                    <span>Linear Regression</span>
                </div>
                <div>
                    Linear regression models the relationship between a dependent variable and one or more 
                    independent variables by fitting a linear equation to observed data. The model assumes 
                    a linear relationship between the input variables (x) and the single output variable (y).
                    
                    Equation: y = β₀ + β₁x₁ + ... + βₙxₙ
                    Where:
                    - y is the predicted value
                    - β₀ is the bias term
                    - β₁...βₙ are the weights for each feature
                    - x₁...xₙ are the input features
                </div>
            </div>
            """