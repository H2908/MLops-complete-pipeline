import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure the log directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
# Setting up logger
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug('Data Loaded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from file: {file_path}, Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error loading data from file: {file_path}, Error: {e}")
        raise e
    
def train_model(X_train, y_train, params):
    try:
        n_estimators = params.get('n_estimators', 100)
        random_state = params.get('random_state', 42)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        logger.debug('Model training completed successfully')
        return clf
    except Exception as e:
        logger.error(f"Error during model training. Error: {e}")
        raise e
    
def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug('Model saved successfully at %s',file_path)
    except Exception as e:
        logger.error(f"Error saving model to file: {file_path}, Error: {e}")
        raise e
def main():
    try:
        params={'n_estimators':25,'random_state':2}
        train_data=load_data('data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        clf=train_model(X_train,y_train,params)
        save_model(clf,'models/random_forest_model.pkl')
    except Exception as e:
        logger.error(f"Error in main execution. Error: {e}")
        raise e
if __name__=='__main__':
    main()