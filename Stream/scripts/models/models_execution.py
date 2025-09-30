from models.scripts.Regression.simple_linear_regression.model_script import model_script , validate_model
from models.scripts.Regression.simple_linear_regression.model_report import model_report
from models.scripts.Regression.simple_linear_regression.model_config import model_config, SLR_model_description , SLR_model_reference_code

# Import 
from models.scripts.hierarchical_clustering.model_config import hierarchy_config, hierarchy_model_description , hierarchy_model_reference_code
from models.scripts.hierarchical_clustering.model_report import hierarchy_report
from models.scripts.hierarchical_clustering.model_script import hierarchy_script,hierarchy_validate_model

# Import KNN functions
from models.scripts.classification.Knn.model_script import model_script as knn_script, validate_model as knn_validate
from models.scripts.classification.Knn.model_report import model_report as knn_report
from models.scripts.classification.Knn.model_config import model_config as knn_config, KNN_model_description , KNN_model_reference_code

# ****************************add your models imports here as that**********************************************************************
# Add this import at the top
from models.scripts.classification.logistic_regression.model_script import model_script as lr_script, validate_model as lr_validate
from models.scripts.classification.logistic_regression.model_report import model_report as lr_report
from models.scripts.classification.logistic_regression.model_config import model_config as lr_config ,LR_model_description , LR_model_reference_code
# *********************************************************************************************************************

def execute_model(model_name,action,param_dict):

    # Example functions for each model
    def run_decision_tree(max_depth, criterion):
        print(f"Running Decision Tree with max_depth={max_depth}, criterion={criterion}")

    def report_decision_tree():
        print("Decision Tree Report")

    def run_knn(n_neighbors):
        print(f"Running KNN with n_neighbors={n_neighbors}")

    def report_knn():
        print("KNN Report")

    def run_svm(kernel, C):
        print(f"Running SVM with kernel={kernel}, C={C}")

    def report_svm():
        print("SVM Report")


    MODELS = {
        "Simple Linear Regression": {
            "script": model_script,
            "report": model_report,
            "config":model_config,
            "validate":validate_model,
            "script_params": ["df", "features", "target", "edit"],
            "config_params":["model_data","edit"],
            "validate_params":["params"]
        },
        # "decision_tree": {
        #     "script": dt_script,
        #     "report": dt_report,
        #     "params": ["data", "features", "target", "max_depth", "min_samples"]
        # }
        "Hierarchical Clustering": {
            "script": hierarchy_script,
            "report": hierarchy_report,
            "config":hierarchy_config,
            "validate":hierarchy_validate_model,
            "script_params": ["df", "features", "n_clusters", "linkage", "metric", "compute_full_tree", "distance_threshold", "edit"],
            "config_params":["model_data","edit"],
            "validate_params":["params"],

        },
        "KNN": {
        "script": knn_script,
        "report": knn_report,
        "config": knn_config,
        "validate": knn_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],

    },
    # ************************************add your models here as that**********************************************************************
    "LR": {
        "script": lr_script,
        "report": lr_report,
        "config": lr_config,
        "validate": lr_validate,
        "script_params": ["df", "features", "target", "edit", "use_grid_search", "param_grid", "manual_params", "cv_folds", "solver", "max_iter"],
        "config_params": ["model_data", "edit"],
        "validate_params": ["params"],

    }
    # *********************************************************************************************************************
        }
    model_info = MODELS[model_name]
    
    # Pick function and expected params
    if action not in model_info:
        raise ValueError(f"Action '{action}' not found for model '{model_name}'")
    func = model_info[action]
    if action=="script":
        expected_params = model_info["script_params"]
    elif action =="config":
        expected_params = model_info['config_params']
        # st.write("config")
    elif action =="validate":
        expected_params = ["params"]
        # model_info['validate_params']
        # st.write("validate")
    elif action == "model_description":
        fun = None
        return  model_info['model_description']
    elif action == "model_reference":
        fun = None
        return model_info['model_reference']

    else : expected_params ={}
    # st.write(param_dict)
    # Extract only the needed params for this model
    args = {k: param_dict[k] for k in expected_params if k in param_dict}

    # st.write(args)
    # Call the model function dynamically
    return func(**args)

def get_model_data(model_name,action):
    MODELS = {
        "Simple Linear Regression": {
            "model_description":SLR_model_description,
            "model_reference":SLR_model_reference_code
        },
        # "decision_tree": {
        #     "script": dt_script,
        #     "report": dt_report,
        #     "params": ["data", "features", "target", "max_depth", "min_samples"]
        # }
        "Hierarchical Clustering": {
            "model_description":hierarchy_model_description,
            "model_reference":  hierarchy_model_reference_code
        },
        "KNN": {
        "model_description":KNN_model_description,
        "model_reference":KNN_model_reference_code
    },
    # ************************************add your models here as that**********************************************************************
    "LR": {
        "model_description":LR_model_description,
        "model_reference":LR_model_reference_code
    }
    
    }
    if model_name not in MODELS:
        return "No description available for this model."
    model_info = MODELS[model_name]
    
    if action == "model_description":
        return  model_info['model_description']
    elif action == "model_reference":
        return model_info['model_reference']