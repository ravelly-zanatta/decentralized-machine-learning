import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pickle

def load_and_split_dataset(dataset_path='dataset.csv', num_nodes=2, node_id=0, is_tester=False, trainer_position=-1):
    try:
        df = pd.read_csv(dataset_path)
        X = df[['medication', 'frequency', 'dose']]
        y = df['target']
    
        # Tratar valores ausentes
        X = X.dropna()
        y = y[X.index]  # garantir alinhamento entre X e y após dropna()

        # Converter variáveis categóricas em variáveis dummy
        X = pd.get_dummies(X, columns=['medication'])
        
        print("numero de festures:", X.shape[1])
        # Separar teste final
        print("Test size:", 1 / num_nodes)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / num_nodes), shuffle=True, stratify=y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=42)
        print("Test len:", len(y_test))
    
        # if is_tester:
        #     X_node_train, X_node_test, y_node_train, y_node_test = [], [], [], []
        # else:
        #     # Split into num_nodes parts
        #     node_size = len(X_train) // (num_nodes - 1)
        #     print("Node index = ", node_id, " node position = ", trainer_position)
        #     print("Node size = ", node_size)
        #     start_idx = trainer_position * node_size
        #     end_idx = (trainer_position + 1) * node_size if trainer_position < num_nodes - 2 else len(X_train)
        #     X_node = X_train.iloc[start_idx:end_idx]
        #     y_node = y_train.iloc[start_idx:end_idx]
        
        if is_tester:
            X_node_train, X_node_test, y_node_train, y_node_test = [], [], [], []
        else:
            print("Node index =", node_id, " | Trainer position =", trainer_position)

            total_size = len(X_train)
            block_size = total_size // 4
            block_idx = trainer_position % 4  
            print(f"Node size: {block_size}")
            start_idx = block_idx * block_size
            end_idx = (block_idx + 1) * block_size if block_idx < 3 else total_size 

            X_node = X_train.iloc[start_idx:end_idx].reset_index(drop=True)
            y_node = y_train.iloc[start_idx:end_idx].reset_index(drop=True)

    
            # Split into train (80%) and test (20%)
            X_node_train, X_node_test, y_node_train, y_node_test = train_test_split(X_node, y_node, test_size=0.2, shuffle=True, stratify=y_node)
        return X_node_train, X_node_test, X_test, y_node_train, y_node_test, y_test
    except Exception as e:
        print(f"Error in load_and_split_dataset: {str(e)}")

def get_model_parameters(model):
    return [model.coef_, model.intercept_]

def set_model_params(model, params):
    model.coef_ = params[0]
    model.intercept_ = params[1]
    return model

def set_initial_params(model):
    model.coef_ = np.zeros((1, 23))  # 3 features: Medicamentos, Dose, Frequencia
    model.intercept_ = np.zeros(1)
    return model

def aggregate_parameters(params_list):
    coef_sum = np.zeros_like(params_list[0][0])
    intercept_sum = np.zeros_like(params_list[0][1])
    for params in params_list:
        coef_sum += params[0]
        intercept_sum += params[1]
    return [coef_sum / len(params_list), intercept_sum / len(params_list)]

def set_model_params_test(model, params):
    model.coef_ = params[0]
    model.intercept_ = params[1]
    model.classes_ = np.array([0, 1])
    return model

