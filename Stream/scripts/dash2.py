import streamlit as st
import plotly.express as px
def run():
# Sample plots
    fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Scatter Plot")
    fig2 = px.bar(x=["A", "B", "C"], y=[4, 3, 7], title="Bar Plot")
    fig3 = px.line(x=[1, 2, 3], y=[7, 6, 2], title="Line Plot")

    # User selects which plots to include
    selected_plots = st.multiselect(
        "Choose plots to add to dashboard",
        options=["Scatter", "Bar", "Line"],
        default=["Scatter", "Bar"]
    )

    # Map names to figures
    plots = {
        "Scatter": fig1,
        "Bar": fig2,
        "Line": fig3
    }

    # Display selected plots in a flexible grid (2 per row)
    rows = [selected_plots[i:i+2] for i in range(0, len(selected_plots), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for i, plot_name in enumerate(row):
            with cols[i]:
                st.plotly_chart(plots[plot_name], use_container_width=True)
