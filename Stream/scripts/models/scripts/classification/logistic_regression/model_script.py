import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds, solver, max_iter):
    try:
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target variable if it's categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_encoder = le
        else:
            target_encoder = None
        
        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            lr = LogisticRegression(random_state=42)
            grid_search = GridSearchCV(
                lr, 
                param_grid, 
                cv=cv_folds, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            st.success(f"Grid Search completed! Best parameters: {best_params}")
            
        else:
            st.info("Training with manual parameters...")
            best_model = LogisticRegression(
                C=manual_params['C'],
                solver=manual_params['solver'],
                max_iter=manual_params['max_iter'],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate AUC-ROC for binary classification
        auc_score = None
        if len(best_model.classes_) == 2:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Store results
        model_results = {
            'model': best_model,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'metrics': {
                'Accuracy': accuracy,
                'Classification Report': class_report,
                'Confusion Matrix': conf_matrix,
                'Best Parameters': best_params,
                'AUC-ROC': auc_score
            },
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'grid_search_params': param_grid if use_grid_search else {},
            'manual_params': manual_params if not use_grid_search else {}
        }
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "c_values", "value": str(param_grid.get('C', [])) if use_grid_search else manual_params['C']},
            {"name": "solver", "value": str(param_grid.get('solver', [])) if use_grid_search else manual_params['solver']},
            {"name": "max_iter", "value": manual_params['max_iter'] if not use_grid_search else str(param_grid.get('max_iter', [100, 200]))},
            {"name": "cv_folds", "value": cv_folds}
        ]
        
        # Create model entry for pipeline
        LR_model = DataManager.create_Logistic_Regression_Model(
            "LogisticRegression",
            param_list,
            st.session_state.selected_trans
        )
        
        # Update pipeline
        if edit:
            # Replace existing Logistic Regression entry
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'LogisticRegression' else LR_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append if no Logistic Regression exists
            st.session_state.pipeline['ML'].append(LR_model)
        
        st.success("Logistic Regression Model created successfully!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training Logistic Regression model: {str(e)}")
        return None

def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target has at least 2 classes
    target_values = params['df'][params['target']].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False
    
    return True