# data_manager.py
import pandas as pd
from typing import Optional

class DataManager:
    _instance = None
    _current_df: Optional[pd.DataFrame] = None
    app_style = """
    <style>
        .transformation-card {
            border-radius: 8px;
            padding: 12px 16px;
            background-color: #010911;
            border-left: 4px solid #4e79a7;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
        .transformation-card:hover {
            background-color: #440032;
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
        .scrollable-list {
                    max-height: 100px;   /* adjust height as you want */
                    overflow-y: auto;
                    padding-right: 10px; /* to avoid overlap with scrollbar */
        }
    </style>
    """
    label_style = """
            <style>
                .custom-container {
                    background-color: #112343;  /* container background */
                    padding: 20px;
                    border-radius: 30px;
                    margin-bottom: 10px;
                    width : 70%;
                }
                .custom-header {
                    color: white;               /* header text color */
                    font-size: 24px;
                    font-weight: bold;
                    margin: 0;                  /* remove default margin */
                }
            </style>
            """

        # Example container with header
    label_data ="""
            <div class="custom-container">
                <h2 class="custom-header">🚀 Dynamic Table with Transformation Pipeline</h2>
            </div>
            """
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    @property
    def current_df(self) -> pd.DataFrame:
        if self._current_df is None:
            raise ValueError("No DataFrame has been loaded yet")
        return self._current_df
    
    @current_df.setter
    def current_df(self, df: pd.DataFrame):
        self._current_df = df.copy()
    
    def get_columns(self, dtype=None) -> list:
        if dtype:
            return [col for col in self.current_df.columns if self.current_df[col].dtype == dtype]
        return list(self.current_df.columns)
    
    def get_sample(self, n=5) -> pd.DataFrame:
        return self.current_df.head(n)
    

        

    #
        # STRUCTURE 
        #  {
        #     "model name":LRM,
        #     "model type" : Linear Regression Model,
        #     "model param" : 
        #     [
        #         {
        #             "name":features,
        #             "value" : age
        #         },
        #         {
        #             "name":target,
        #             "value" : Blood Pressure
        #         },
        #     ]
        
        # } 
    #
    @staticmethod
    def create_Linear_REG_Model(model_name, model_param,trans) -> dict:
        return {
            "model name": model_name,
            "model type": "Simple Linear Regression",
            "model param": model_param,
            "transformations" : trans
        }
    

    @staticmethod
    def create_KNN_Model(model_name, model_param, trans):
        # Ensure proper parameter types when creating the model
        processed_params = []
        for param in model_param:
            # Convert numeric values to appropriate types
            if param['name'] in ['n_neighbors_range', 'cv_folds']:
                try:
                    if param['name'] == 'cv_folds':
                        param['value'] = int(param['value'])
                    elif isinstance(param['value'], str) and ',' in param['value']:
                        # It's a list, keep as string representation
                        pass
                    else:
                        param['value'] = int(param['value'])
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "KNN",
            "model param": processed_params,
            "transformations": trans
        }
    
# **************************************add this part**************************************************************************
    @staticmethod
    def create_Logistic_Regression_Model(model_name, model_param, trans):
        # Ensure proper parameter types
        processed_params = []
        for param in model_param:
            # Handle max_iter conversion
            if param['name'] == 'max_iter' and isinstance(param['value'], str) and param['value'].startswith('['):
                try:
                    import re
                    numbers = re.findall(r'\d+', param['value'])
                    param['value'] = int(numbers[0]) if numbers else 100
                except (ValueError, TypeError):
                    param['value'] = 100
            processed_params.append(param)
        
        return {
            "model name": model_name,
            "model type": "Logistic Regression Classifier",
            "model param": processed_params,
            "transformations": trans
        }
    # ****************************************************************************************************************