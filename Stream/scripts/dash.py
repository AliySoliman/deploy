import streamlit as st
import random

# CSS styling
st.markdown("""
<style>
    .transformation-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        background-color: #f8f9fa;
        border-left: 4px solid #4e79a7;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .transformation-card:hover {
        background-color: #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .initials-badge {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        margin-right: 12px;
        flex-shrink: 0;
    }
    .card-content {
        flex-grow: 1;
        min-width: 0;
    }
    .card-title {
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-type {
        font-size: 0.8em;
        color: #6c757d;
    }
    .action-button {
        background: none;
        border: none;
        padding: 6px;
        margin: 0 4px;
        cursor: pointer;
        font-size: 16px;
        opacity: 0.7;
        transition: all 0.2s ease;
    }
    .action-button:hover {
        opacity: 1;
        transform: scale(1.1);
    }
    .edit-button {
        color: #4e79a7;
    }
    .delete-button {
        color: #e15759;
    }
    .container-border {
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def get_initials(name):
    """Get initials from transformation name"""
    words = name.split()
    if len(words) >= 2:
        return f"{words[0][0]}{words[-1][0]}".upper()
    return name[:2].upper() if len(name) >= 2 else name[0].upper() * 2

def get_color_from_type(transformation_type):
    """Get a consistent color based on transformation type"""
    colors = {
        "delete": "#e15759",
        "computation": "#4e79a7", 
        "binning": "#59a14f",
        "modify": "#edc948",
        "categorize": "#af7aa1",
        "default": "#76b7b2"
    }
    return colors.get(transformation_type.lower(), colors["default"])

# Example usage
if "pipeline" not in st.session_state:
    st.session_state.pipeline = [
        {"name": "Gender Delete", "type": "delete"},
        {"name": "Total Score Calculation", "type": "computation"},
        {"name": "Age Binning", "type": "binning"},
        {"name": "Name Formatting", "type": "modify"},
        {"name": "Pass/Fail Categorization", "type": "categorize"}
    ]

with st.container():
    st.markdown('<div class="container-border">', unsafe_allow_html=True)
    
    for i, step in enumerate(st.session_state.pipeline):
        col1, col2 = st.columns([5, 1])
        with col1:
            initials = get_initials(step["name"])
            color = get_color_from_type(step["type"])
            
            st.markdown(f"""
            <div class="transformation-card" title="{step['name']} ({step['type']})">
                <div class="initials-badge" style="background-color: {color}">
                    {initials}
                </div>
                <div class="card-content">
                    <div class="card-title">{step['name']}</div>
                    <div class="card-type">{step['type']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Edit button
            st.markdown(f"""
            <button class="action-button edit-button" onclick="
                const event = new CustomEvent('editTransform', {{ detail: {{ index: {i} }} }});
                document.dispatchEvent(event);
            ">✏️</button>
            """, unsafe_allow_html=True)
            
            # Delete button
            st.markdown(f"""
            <button class="action-button delete-button" onclick="
                const event = new CustomEvent('deleteTransform', {{ detail: {{ index: {i} }} }});
                document.dispatchEvent(event);
            ">🗑️</button>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# JavaScript handlers
st.components.v1.html("""
<script>
    document.addEventListener('editTransform', function(e) {
        const index = e.detail.index;
        Streamlit.setComponentValue({type: "edit", index: index});
    });
    
    document.addEventListener('deleteTransform', function(e) {
        const index = e.detail.index;
        Streamlit.setComponentValue({type: "delete", index: index});
    });
</script>
""", height=0)

# Handle the actions
if st.session_state.get("component_value"):
    action = st.session_state.component_value
    if action["type"] == "edit":
        st.session_state.editing_transformation = st.session_state.pipeline[action["index"]]
        st.session_state.show_modal = True
        st.rerun()
    elif action["type"] == "delete":
        st.session_state.pipeline.pop(action["index"])
        st.rerun()