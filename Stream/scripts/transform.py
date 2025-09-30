import streamlit as st
import pandas as pd
import re
from streamlit_elements import elements, mui, html
import save
from constants import DataManager
from collections import defaultdict


data_manager = DataManager()

def run():
    # Initialize session state
    if 'load' not in st.session_state :
        st.session_state.load =True
    if 'save_load' not in st.session_state :
        st.session_state.save_load =False
    if 'edit' not in st.session_state:
        st.session_state.edit = False
    if 'editing_transformation' not in st.session_state:
        st.session_state.editing_transformation = None
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "pipeline" not in st.session_state:
            st.session_state.pipeline = {}

    if "transformations" not in st.session_state.pipeline:
        st.session_state.pipeline["transformations"] = []   
        st.session_state.pipeline.setdefault("ML", [])

   
    if "transformed_dfs" not in st.session_state:
        st.session_state.transformed_dfs = {"original": None}
    if "df_to_show" not in st.session_state:
        st.session_state.df_to_show = pd.DataFrame()

    # UI Layout
    st.markdown(data_manager.label_style, unsafe_allow_html=True)
    st.markdown(data_manager.label_data, unsafe_allow_html=True)

    # st.title("Dynamic Table with Transformation Pipeline")
    # Upload dataset
    data_col , save_col,submit_col = st.columns([4,1,1])
    with data_col:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    with save_col :
            save_file = st.text_input("save file path No quotations",key = 'save_upload')
            if st.button('upload'):
                temp = load_data(save_file)
                st.session_state.save_file_upload = save_file
                # st.session_state.save_upload=""
                # st.experimental_rerun()         # force UI refresh

                if temp :
                    st.session_state.pipeline = load_data(save_file) 
                else :
                    st.session_state.pipeline.setdefault("transformations", [])
                # current_pipeline = st.session_state.pipeline


    if uploaded_file:
        st.session_state.df_original = pd.read_csv(uploaded_file)
        data_manager.current_df = st.session_state.df_original
        st.session_state.transformed_dfs["original"] = st.session_state.df_original.copy()
        # if st.session_state.load:
        
        st.session_state.load = False

        st.write(st.session_state.pipeline)


    # list_changed = current_pipeline != st.session_state.pipeline
    # Display transformations selector and table
    
    display_transformations_and_table()
   
        # Transformation modal
    if st.button("Add Transformation"):
            st.session_state.show_modal = True
            st.session_state.editing_transformation = None
            st.rerun()


    if st.session_state.get("show_modal", False):
        display_transformation_modal()

def display_transformations_and_table():
    """Display the transformations selector and resulting table"""
    if st.session_state.df_original is not None and st.session_state.pipeline:
        # Select which transformations to apply
        selected_steps = st.multiselect(
            "Choose pipeline steps to view result", 
            ["original"] + [step["name"] for step in st.session_state.pipeline['transformations']]
        )
        
        # Apply selected transformations
        st.session_state.df_to_show = apply_selected_transformations(selected_steps)

        
        # Display table and transformations list side by side

        st.dataframe(st.session_state.df_to_show)
        with st.sidebar:
                st.markdown(
                    """
                    <style>
    <style>
        /* Define color variables */
        :root {
            --oxford-blue: #112343;
            --oxford-blue-2: #0f1533;
            --indigo-dye: #123b5f;
            --violet-blue: #3941aa;
            --vista-blue: #7b99c4;
        }

        /* Example: change sidebar background */
        [data-testid="stSidebar"] {
            background: var(--gradient-right);
            color: white;
        }

        /* Example: add gradient background */
        body {
            background: linear-gradient(135deg, var(--oxford-blue), var(--indigo-dye), var(--vista-blue));
        }


        /* Define gradients */
        :root {
            --gradient-right: linear-gradient(90deg, #0a0f20, #0d1230, #0e2440, #2c3275, #5a7290);
        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.header("🛠️ Transformations")
                st.markdown('<div class="custom-container">', unsafe_allow_html=True)
                display_transformations_list()

def apply_selected_transformations(selected_steps):
    """Apply the selected transformations to the original dataframe"""
    df_to_show = st.session_state.df_original.copy()
    
    for step in st.session_state.pipeline['transformations']:
        if step["name"] in selected_steps:
            if step["type"] == "delete":
                df_to_show.drop(columns=[step["column"]], inplace=True)
            elif step["type"] == "computation":
                df_to_show[step["new_column"]] = calculate_expression(df_to_show, step['expr'])
            elif step["type"] == "filter":
                df_to_show = df_to_show[df_to_show[step["column"]] == step["value"]]
            elif step["type"] == "group":
                df_to_show = group_and_aggregate(df_to_show,step["group_col"], step["target_col"], step["agg_choice"])
    return df_to_show

def display_transformations_list():
    st.markdown(data_manager.app_style, unsafe_allow_html=True)

    if st.session_state.pipeline:
        # Create a container with scrollable CSS

            for i, step in enumerate(st.session_state.pipeline['transformations']):
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
                    if st.button("✏️", key=f"edit_{i}"):
                        st.session_state.editing_transformation = step
                        st.session_state.show_modal = True
                        st.rerun()
                        
                    if st.button("🗑️", key=f"delete_{i}",type= 'primary'):
                        st.session_state.pipeline['transformations'].pop(i)
                        st.rerun()

            # Close scrollable div



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






def display_transformation_modal():
    """Display the modal for adding/editing transformations"""
    with st.container():
        st.header("Add Transformation" if not st.session_state.editing_transformation else "Edit Transformation")
        
        # Set default values for editing
        edit_values = st.session_state.editing_transformation if st.session_state.editing_transformation else {}
        
        # Transformation name
        transform_name = st.text_input(
            "Transformation Name",
            value=edit_values.get("name", "")
        )
        
        # Transformation type selector
        transformations = ["delete", "computation", "filter","group"]
        transform_type = st.selectbox(
            "Transformation Type",
            transformations,
            index=
             transformations.index(edit_values.get("type", "delete"))
            )

        with st.expander("Transformation Settings", expanded=True):
        # Display appropriate fields based on transformation type
            transformation_params = {}
            if st.session_state.df_original is not None:
                if transform_type == "delete":
                    transformation_params = build_delete_transf(
                        df=st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif transform_type == "computation":
                    transformation_params = build_computation_transf(
                        df=st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif transform_type == "filter":
                    transformation_params = build_filter_transf(
                        df=st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif transform_type == "group":
                    transformation_params = build_group_transf(
                        df=st.session_state.df_original,
                        edit_values=edit_values
                    )

        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Apply"):
                save_transformation(
                    transform_name,
                    transform_type,
                    transformation_params,
                    is_edit=bool(st.session_state.editing_transformation))
        with col2:
            if st.button("Cancel"):
                st.session_state.show_modal = False
                st.rerun()

def build_delete_transf(df, edit_values=None) -> dict:
    """Build the delete transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_delete = st.selectbox(
        "Column to delete",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    return {"column": col_to_delete}

def build_computation_transf(df, edit_values=None) -> dict:
    """Build the computation transformation UI"""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.markdown(f"**Numeric Columns:** {' | '.join(numeric_cols)}")
    
    default_expr = edit_values.get("expr", "") if edit_values else ""
    expr_cols = st.text_input("Expression", value=default_expr)
    
    default_new_col = edit_values.get("new_column", "") if edit_values else ""
    new_col_name = st.text_input("New column name", value=default_new_col)
    
    return {"expr": expr_cols, "new_column": new_col_name}

def build_filter_transf(df, edit_values=None) -> dict:
    """Build the filter transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_filter = st.selectbox(
        "Column to filter",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    
    default_value = edit_values.get("value", "") if edit_values else ""
    value_to_filter = st.text_input("Value to filter by", value=default_value)
    
    return {"column": col_to_filter, "value": value_to_filter}

def save_transformation(name, type, params, is_edit=False):
    """Save the transformation to the pipeline"""
    new_step = {"name": name, "type": type, **params}
    
    if is_edit:
        # Find and replace the existing transformation
        for i, step in enumerate(st.session_state.pipeline['transformations']):
            if step["name"] == st.session_state.editing_transformation["name"]:
                st.session_state.pipeline['transformations'][i] = new_step
                break
    else:
        # Add new transformation
        st.session_state.pipeline['transformations'].append(new_step)
    
    st.session_state.show_modal = False
    st.session_state.editing_transformation = None
    st.rerun()

def calculate_expression(df, expr: str) -> pd.Series:
    """Evaluates a mathematical expression string with column names in brackets"""
    columns = re.findall(r'\[([^\[\]]+)\]', expr)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        expr = expr.replace(f"[{col}]", f"df['{col}']")
    
    try:
        return eval(expr)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")
def load_data(f):
  
    loaded_data = save.load_encoded_transformations(f
        #  "D:/WAHBA/Projects/Stream/save files/transformations.enc"
        )
    return loaded_data
    import streamlit as st
import pandas as pd
def build_group_transf(df, edit_values=None) -> dict:
    """Build the group transformation UI"""
    default_group_col = edit_values.get("group_col", df.columns[0]) if edit_values else df.columns[0]
    group_col = st.selectbox(
        "Column to group by",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_group_col) if edit_values else 0
    )
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    default_target_col = edit_values.get("target_col", numeric_cols[0]) if edit_values else numeric_cols[0]
    target_col = st.selectbox(
        "Numeric column to aggregate",
        numeric_cols,
        index=numeric_cols.index(default_target_col) if edit_values else 0
    )
    
    agg_functions = ['sum', 'mean', 'max', 'min', 'count']
    default_agg = edit_values.get("agg_choice", agg_functions[0]) if edit_values else agg_functions[0]
    agg_choice = st.selectbox(
        "Aggregation function",
        agg_functions,
        index=agg_functions.index(default_agg) if edit_values else 0
    )
    
    return {"group_col": group_col, "target_col": target_col, "agg_choice": agg_choice}
def group_and_aggregate(df: pd.DataFrame, group_col: str, target_col: str, agg_choice: str) -> pd.DataFrame:

    # Perform groupby
    grouped_df = (
        df.groupby(group_col)[target_col]
        .agg(agg_choice)
        .reset_index()
        .rename(columns={target_col: f"{agg_choice}_{target_col}"})
    )

    st.write("### 📊 Aggregated Data")
    # st.dataframe(grouped_df)

    return grouped_df
