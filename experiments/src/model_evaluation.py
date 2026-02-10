import os
import numpy as np
import pandas as pd
import pickle
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug('Model loaded successfully from %s', file_path)
        return model
    except Exception as e:
        logger.error(
            f"Error loading model from file: {file_path}, Error: {e}"
        )
        raise e


def evalute_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)

        # FIX: get probability for 'spam' class explicitly
        spam_index = list(clf.classes_).index('spam')
        y_pred_prob = clf.predict_proba(X_test)[:, spam_index]

        accuracy = accuracy_score(y_test, y_pred)

        # FIX: specify pos_label
        precision = precision_score(
            y_test, y_pred, pos_label='spam'
        )
        recall = recall_score(
            y_test, y_pred, pos_label='spam'
        )

        roc_auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug(
            'Model evaluation completed successfully with metrics: %s',
            metrics_dict
        )
        return metrics_dict

    except Exception as e:
        logger.error(
            f"Error during model evaluation. Error: {e}"
        )
        raise e


def save_metrics(metrics_dict: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.debug('Metrics saved successfully at %s', file_path)
    except Exception as e:
        logger.error(
            f"Error saving metrics to file: {file_path}, Error: {e}"
        )
        raise e


def main():
    try:
        model = load_model('models/random_forest_model.pkl')

        test_data = pd.read_csv('data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics_dict = evalute_model(model, X_test, y_test)

        save_metrics(
            metrics_dict,
            'metrics/model_metrics.json'
        )

    except Exception as e:
        logger.error(
            f"Error in main execution. Error: {e}"
        )
        print(f"Error in main execution. Error: {e}")


if __name__ == '__main__':
    main()
