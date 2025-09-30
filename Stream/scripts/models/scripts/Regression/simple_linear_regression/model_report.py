import streamlit as st
def model_report():
    results = st.session_state.model_results
    st.markdown("""
                        <div class="report-container">
                            <h3>Model Performance Report</h3>
                            <div class="metric-card">
                                <strong>Features:</strong> {features}<br>
                                <strong>Target:</strong> {target}
                            </div>
                            <div class="metric-card">
                                <strong>Mean Squared Error:</strong> {mse:.4f}<br>
                                <strong>R² Score:</strong> {r2:.4f}
                            </div>
                            <div class="metric-card">
                                <strong>Coefficients:</strong><br>
                                {coeffs}
                            </div>
                            <div class="metric-card">
                                <strong>Intercept:</strong> {intercept:.4f}
                            </div>
                        </div>
                        """.format(
                            features=", ".join(results['features']),
                            target=results['target'],
                            mse=results['metrics']['MSE'],
                            r2=results['metrics']['R2 Score'],
                            coeffs="<br>".join([f"&nbsp;&nbsp;{feat}: {coef:.4f}" 
                                            for feat, coef in zip(results['features'], results['metrics']['Coefficients'])]),
                            intercept=results['metrics']['Intercept']
                        ), unsafe_allow_html=True)
                        # st.write(st.session_state.model_data)