import json
from base64 import b64encode, b64decode
import zlib
import streamlit as st
import os 

def save_encoded_transformations( transformations_dict):
    """
    Save multiple transformations lists to an encoded JSON file with keys.
    
    Args:
        file_path (str): Path to save the file
        transformations_dict (dict): Dictionary of transformations lists with keys
    """
    # Convert to JSON string
    json_str = json.dumps(transformations_dict)
    
    # Compress and encode
    compressed = zlib.compress(json_str.encode('utf-8'))
    encoded = b64encode(compressed)
    if "save_file_upload" not in st.session_state:
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        fpath = os.path.join(downloads_dir, "save.enc")
    else :  
        fpath = st.session_state.save_file_upload
        # Write to file
    with open(fpath, 'wb') as f:
        f.write(encoded)
    st.success(f'Your Progress has been saved to {fpath}', icon="✅")


def load_encoded_transformations(file_path):
    """
    Load and decode transformations lists from an encoded file.
    
    Args:
        file_path (str): Path to the encoded file
        
    Returns:
        dict: Dictionary of transformations lists with their keys
    """
    json_str=""
    # Read encoded data
    try:
     with open(file_path, 'rb') as f:
        encoded = f.read().decode("utf-8")
    
    # Decode and decompress
        compressed = b64decode(encoded)
        json_str = zlib.decompress(compressed).decode('utf-8')
    except FileNotFoundError as e:
        st.write("No previous transformations found ")
    # Convert back to Python objects
    return json.loads(json_str) if json_str !="" else ""

def get_transformations_by_key(loaded_data, key):
    """
    Get a specific transformations list by its key.
    
    Args:
        loaded_data (dict): Dictionary returned by load_encoded_transformations
        key (str): Key of the transformations list to retrieve
        
    Returns:
        list: The requested transformations list or None if not found
    """
    return loaded_data.get(key)

# Example usage
if __name__ == "__main__":
    # Create a dictionary with named transformations lists
    transformations_data = {
        "student_grades": [
            {"name": "gender delete", "type": "delete", "column": "gender"},
            {"name": "Total Score", "type": "computation", "expr": "[math score] + [reading score] * 2", "new_column": ""}
        ],
        "student_demographics": [
            {"name": "age bin", "type": "binning", "column": "age", "bins": [0, 18, 35, 60, 100]},
            {"name": "score avg", "type": "computation", "expr": "([math score] + [reading score]) / 2", "new_column": "average score"}
        ],
        "student_processing": [
            {"name": "name uppercase", "type": "modify", "column": "name", "operation": "upper"},
            {"name": "pass/fail", "type": "categorize", "column": "average score", "threshold": 60}
        ]
    }
    
    # Save to file
    save_encoded_transformations("transformations.enc", transformations_data)
    
    # Load from file
    loaded_data = load_encoded_transformations("transformations.enc")
    
    # Retrieve specific lists by their keys
    grades_transformations = get_transformations_by_key(loaded_data, "student_grades")
    demo_transformations = get_transformations_by_key(loaded_data, "student_demographics")
    processing_transformations = get_transformations_by_key(loaded_data, "student_processing")
    
    print("Student Grades Transformations:", grades_transformations)
    print("Student Demographics Transformations:", demo_transformations)
    print("Student Processing Transformations:", processing_transformations)