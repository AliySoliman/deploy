import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import transform
def run():
        if not st.session_state.load:
            # st.write(st.session_state.pipeline)
            if 'report_df' not in st.session_state:
                st.session_state.report_df = st.session_state.df_original
            selected_trans = st.multiselect(
                "transformations",
                options=["original"] + [step["name"] for step in st.session_state.pipeline['transformations']],
            )
            st.session_state.report_df = transform.apply_selected_transformations(selected_trans)

            # Sample DataFrame


            st.title("ML Model Report")

            # 1️⃣ Show DataFrame
            st.subheader("Dataset")
            st.dataframe(st.session_state.report_df)
            # Model info
            model_name = st.text_input("Model Name", "Random Forest Classifier")
            features = list(st.session_state.report_df.columns[:-1])
            target = st.session_state.report_df.columns[-1]
            comments = st.text_area("Comments", "This is a sample comment.")

            # Show extracted info
            st.write(f"**Features:** {features}")
            st.write(f"**Target:** {target}")

            # PDF generation
            def create_pdf(filename, model_name, features, target, comments, df):
                doc = SimpleDocTemplate(filename, pagesize=letter)
                styles = getSampleStyleSheet()
                content = []

                # Add text
                content.append(Paragraph(f"Model Name: {model_name}", styles['Heading1']))
                content.append(Paragraph(f"Features: {', '.join(features)}", styles['Normal']))
                content.append(Paragraph(f"Target: {target}", styles['Normal']))
                content.append(Spacer(1, 12))
                content.append(Paragraph("Comments:", styles['Heading2']))
                content.append(Paragraph(comments, styles['Normal']))
                content.append(Spacer(1, 12))

                # Add table for DataFrame
                table_data = [df.columns.tolist()] + df.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                content.append(table)

                doc.build(content)

            # Button
            if st.button("Generate PDF"):
                pdf_filename = "model_report.pdf"
                create_pdf(pdf_filename, model_name, features, target, comments, st.session_state.report_df.head(20))
                st.success("PDF Generated!")
                with open(pdf_filename, "rb") as file:
                    st.download_button("Download PDF", file, file_name=pdf_filename)