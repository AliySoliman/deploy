import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def model_report():
    if 'model_results' not in st.session_state:
        st.error("No model results found. Please create a model first.")
        return
    
    results = st.session_state.model_results
    
    # Check if results contain classification metrics
    if 'Accuracy' not in results['metrics']:
        st.error("Invalid model results format for Logistic Regression classifier")
        return
    
    # Extract metrics from classification report
    class_report = results['metrics']['Classification Report']
    
    # Calculate macro averages for Precision, Recall, F1-Score
    precision_avg = class_report.get('macro avg', {}).get('precision', 0)
    recall_avg = class_report.get('macro avg', {}).get('recall', 0)
    f1_avg = class_report.get('macro avg', {}).get('f1-score', 0)
    accuracy = results['metrics']['Accuracy']
    auc_score = results['metrics'].get('AUC-ROC', 'N/A')
    
    st.markdown("""
    <div class="report-container">
        <h3>Logistic Regression Classifier Performance Report</h3>
        <div class="metric-card">
            <strong>Features:</strong> {features}<br>
            <strong>Target:</strong> {target}<br>
            <strong>Grid Search Used:</strong> {grid_search}
        </div>
        <div class="metric-card">
            <strong>Accuracy:</strong> {accuracy:.4f}<br>
            <strong>Precision (Macro Avg):</strong> {precision:.4f}<br>
            <strong>Recall (Macro Avg):</strong> {recall:.4f}<br>
            <strong>F1-Score (Macro Avg):</strong> {f1:.4f}<br>
            <strong>AUC-ROC:</strong> {auc}
        </div>
        <div class="metric-card">
            <strong>Best Parameters:</strong> {best_params}
        </div>
    </div>
    """.format(
        features=", ".join(results['features']),
        target=results['target'],
        grid_search="Yes" if results['use_grid_search'] else "No",
        accuracy=accuracy,
        precision=precision_avg,
        recall=recall_avg,
        f1=f1_avg,
        auc=auc_score if auc_score != 'N/A' else f"{auc_score:.4f}",
        best_params=results['metrics']['Best Parameters']
    ), unsafe_allow_html=True)
    
    # Detailed Classification Report Section
    st.subheader("📊 Detailed Classification Report")
    
    # Convert classification report to DataFrame for better display
    class_df = pd.DataFrame(class_report).transpose().round(4)
    st.dataframe(class_df, use_container_width=True)
    
    # Metrics Visualization
    st.subheader("📈 Metrics Visualization")
    
    # Create a bar chart for main metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # ROC Curve for binary classification
    if auc_score != 'N/A' and results['model'] and hasattr(results['model'], 'predict_proba'):
        st.subheader("📊 ROC Curve")
        y_test = results.get('y_test')
        if y_test is not None:
            y_pred_proba = results['model'].predict_proba(results['X_test'])[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
    
    # Confusion Matrix Visualization
    st.subheader("🎯 Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(results['metrics']['Confusion Matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=True, yticklabels=True)
    
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    ax2.set_title('Confusion Matrix')
    st.pyplot(fig2)
    
    # Model Configuration Details
    st.subheader("⚙️ Model Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Hyperparameters:**")
        best_params = results['metrics']['Best Parameters']
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")
        
        st.write("**Training Method:**")
        st.write(f"- Grid Search: {'✅ Yes' if results['use_grid_search'] else '❌ No'}")
    
    with col4:
        st.write("**Dataset Information:**")
        st.write(f"- Number of features: **{len(results['features'])}**")
        st.write(f"- Target variable: **{results['target']}**")
        
        # Safely get number of classes
        num_classes = "N/A"
        if 'target_encoder' in results and results['target_encoder'] is not None:
            num_classes = len(results['target_encoder'].classes_)
        elif 'Confusion Matrix' in results['metrics']:
            num_classes = results['metrics']['Confusion Matrix'].shape[0]
        
        st.write(f"- Classes: **{num_classes}**")
        st.write(f"- AUC-ROC: **{auc_score if auc_score != 'N/A' else f'{auc_score:.4f}'}**")