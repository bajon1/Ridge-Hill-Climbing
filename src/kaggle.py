import os
import pickle
import pandas as pd
import numpy as np

data = pd.read_csv("../dataset/train.csv")

target = "Heart Disease"
data['Heart Disease'] = np.where(data['Heart Disease'] == 'Presence', 1, 0)

data.drop('id', axis=1, inplace=True)
categorical_cols = data.columns[(data.nunique() <= 10) & (data.nunique() > 2)]
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=float)

all_cols = data_encoded.columns.tolist()

MI_cols = ['Chest pain type_3', 'EKG results_2', 'Number of vessels fluro_2',
    'Number of vessels fluro_1', 'Age', 'Chest pain type_2',
    'Number of vessels fluro_3', 'Cholesterol', 'BP', 'Slope of ST_3',
    'FBS over 120', 'Thallium_6', 'EKG results_1']

RFECV_cols = ['Chest pain type_2', 'Chest pain type_3', 'Chest pain type_4',
       'EKG results_1', 'EKG results_2', 'Slope of ST_2', 'Slope of ST_3',
       'Number of vessels fluro_1', 'Number of vessels fluro_2',
       'Number of vessels fluro_3', 'Thallium_6', 'Thallium_7', 'Sex',
       'FBS over 120', 'Exercise angina', 'Cholesterol', 'ST depression',
       'Age', 'Max HR']

boruta_cols = ['Chest pain type_4', 'Slope of ST_2', 'Thallium_7', 'Exercise angina',
        'Cholesterol', 'ST depression', 'Age', 'Max HR']

def create_kaggle_submission(model_name,models_dir="models1", test_path="../dataset/test.csv",
                             sample_submission_path="../dataset/sample_submission.csv", output_dir="submissions"):

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_name}")

    test = pd.read_csv(test_path)

    ids = test["id"]

    test = test.drop(columns=["id"])

    categorical_cols = test.columns[(test.nunique() <= 10) & (test.nunique() > 2)]
    test_encoded = pd.get_dummies(test, columns=categorical_cols, drop_first=True, dtype=float)

    model_features = model.feature_names_in_

    for col in model_features:
        if col not in test_encoded.columns:
            test_encoded[col] = 0

    test_encoded = test_encoded[model_features]

    proba = model.predict_proba(test_encoded)[:, 1]

    submission = pd.read_csv(sample_submission_path)

    submission["Heart Disease"] = proba
    submission["id"] = ids

    output_path = os.path.join(output_dir, f"submission_{model_name}.csv")
    submission.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")

    return submission

create_kaggle_submission("RF_ALL")