import tkinter as tk
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the model
model = pickle.load(open('Obesity/rf_model.pkl', 'rb'))

preprocessor = pickle.load(open('Obesity/fitted_preprocessor.pkl', 'rb'))

# # Load the scaler
# scaler = pickle.load(open('Obesity/scaler.pkl', 'rb'))

window = tk.Tk()
window.title("Klasifikasi")
window.geometry("400x400")

label = tk.Label(window, text="Klasifikasi", font=("Arial", 16))
label.pack()

input_labels = [
    'Gender',
    'Age',
    'Siblings/Spouses Aboard',
    'Parents/Children Aboard',
    'Fare',
    'Embarked from',
]

inputs = {
    'Sex': ('dropdown',['male', 'female']),
    'Age': ('entry', None),
    'SibSp': ('entry', None),
    'Parch': ('entry', None),
    'Fare': ('entry', None),
    'Embarked': ('dropdown', ['S', 'C', 'Q']),
}

input_vars = {}

for label_text, key in zip(input_labels, inputs.keys()):
    # For Label, use anchor='w' to align text to the left and pack with anchor='w' to align the widget to the left
    tk.Label(window, text=label_text, anchor='w').pack(fill='x', anchor='w')
    input_type, options = inputs[key]
    if input_type == 'dropdown':
        var = tk.StringVar(window)
        var.set(options[0])  # default value
        # Align OptionMenu to the left
        tk.OptionMenu(window, var, *options).pack(anchor='w')
    elif input_type == 'entry':
        var = tk.StringVar(window)
        # For Entry, use justify='left' to align text to the left and pack with anchor='w' to align the widget to the left
        tk.Entry(window, textvariable=var, justify='left').pack(anchor='w')
    input_vars[key] = var



numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])



def transform_input(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ])
    return preprocessor.fit_transform(df)


def mapper(df):
    # Create 'Gender_female' and 'Gender_male' columns
    df['Sex_female'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
    df['Sex_male'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    return df



def predict():
    input_data = {key: [var.get()] for key, var in input_vars.items()}
    df = pd.DataFrame(input_data)

    # Apply preprocessor to df
    df = preprocessor.transform(df)

    prediction = model.predict(df)

    # Convert prediction to string and update the prediction_label
    prediction_text = f"Prediction: {prediction[0]}"
    prediction_label.config(text=prediction_text)

button = tk.Button(window, text="Submit", command=predict)
button.pack(anchor='w')

prediction_label = tk.Label(window, text="", anchor='w')
prediction_label.pack(fill='x', anchor='w')
window.mainloop()