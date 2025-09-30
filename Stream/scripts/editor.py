
from streamlit_elements import elements, mui, html ,sync,lazy
import streamlit as st
def run():
    with elements("monaco_editors"):

        # Streamlit Elements embeds Monaco code and diff editor that powers Visual Studio Code.
        # You can configure editor's behavior and features with the 'options' parameter.
        #
        # Streamlit Elements uses an unofficial React implementation (GitHub links below for
        # documentation).

        from streamlit_elements import editor

        if "content" not in st.session_state:
            st.session_state.content = "Default value"

        mui.Typography("Content: ", st.session_state.content)

        def update_content(value):
            st.session_state.content = value

        editor.Monaco(
            height=300,
            defaultValue=st.session_state.content,
            onChange=lazy(update_content)
        )

        mui.Button("Update content", onClick=sync())

        editor.MonacoDiff(
            original="Happy Streamlit-ing!",
            modified="Happy Streamlit-in' with Elements!",
            height=300,
        )