import tkinter as tk
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('Obesity/svc_model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('Obesity/scaler.pkl', 'rb'))

window = tk.Tk()
window.title("Obesity Classification")
window.geometry("800x1000")

label = tk.Label(window, text="Obesity Classification", font=("Arial", 16))
label.pack()

input_labels = [
    'Gender',
    'Age',
    'Height (m)',
    'Weight (kg)',
    'Has a family member suffered or suffers from overweight?	',
    'Do you eat high caloric food frequently? (1-3)',
    'Do you usually eat vegetables in your meals? (1-4)',
    'How many main meals do you have daily?',
    'Do you eat any food between meals?',
    'Do you smoke?',
    'How much water do you drink daily? (1-3 Liters)',
    'Do you monitor the calories you eat daily?',
    'How often do you have physical activity? (0-3)',
    'How much time do you use technological devices such as cell phone, videogames, television, computer and others? (0-2)',
    'How often do you drink alcohol?',
    'Which transportation do you usually use?'
]

inputs = {
    'Gender': ('dropdown',['Male', 'Female']),
    'Age': ('entry', None),
    'Height': ('entry', None),
    'Weight': ('entry', None),
    'family_history_with_overweight': ('dropdown', ['Yes', 'No']),
    'FAVC': ('dropdown', ['Yes', 'No']),
    'FCVC': ('entry', None),
    'NCP': ('entry', None),
    'CAEC': ('dropdown', ['No', 'Sometimes', 'Frequently', 'Always']),
    'SMOKE': ('dropdown', ['Yes', 'No']),
    'CH2O': ('entry', None),
    'SCC': ('dropdown', ['Yes', 'No']),
    'FAF': ('entry', None),
    'TUE': ('entry', None),
    'CALC': ('dropdown', ['No', 'Sometimes', 'Frequently', 'Always']),
    'MTRANS': ('dropdown', ['Bike', 'Motorbike', 'PublicTransportation', 'Automobile', 'Walking'])
}

input_vars = {}

for label_text, key in zip(input_labels, inputs.keys()):
    tk.Label(window, text=label_text).pack()
    input_type, options = inputs[key]
    if input_type == 'dropdown':
        var = tk.StringVar(window)
        var.set(options[0])  # default value
        tk.OptionMenu(window, var, *options).pack()
    elif input_type == 'entry':
        var = tk.StringVar(window)
        tk.Entry(window, textvariable=var).pack()
    input_vars[key] = var



def map_transportation(value):
    mapping = {
        'PublicTransportation': 3,
        'Automobile': 0,
        'Walking': 4,
        'Motorbike': 2,
        'Bike': 1
    }
    return mapping.get(value, value)

def map_calc(value):
    mapping = {
        'Sometimes': 2,
        'No': 3,
        'Frequently': 1,
        'Always': 0
    }
    return mapping.get(value, value)

def map_scc(value):
    mapping = {
        'No': 0,
        'Yes': 1
    }
    return mapping.get(value, value)

def map_gender(value):
    mapping = {
        'Male': 1,
        'Female': 0
    }
    return mapping.get(value, value)

def map_pred(value):
    mapping = {
        2: 'Obesity_Type_I',
        4: 'Obesity_Type_III',
        3: 'Obesity_Type_II',
        5: 'Overweight_Level_I',
        6: 'Overweight_Level_II',
        1: 'Normal_Weight',
        0: 'Insufficient_Weight'
    }
    return mapping.get(value, value)


def predict():
    input_data = {key: [var.get()] for key, var in input_vars.items()}
    df = pd.DataFrame(input_data)


    df['Gender'] = df['Gender'].apply(map_gender)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].apply(map_scc)
    df['FAVC'] = df['FAVC'].apply(map_scc)
    df['CAEC'] = df['CAEC'].apply(map_calc)
    df['SMOKE'] = df['SMOKE'].apply(map_scc)
    df['SCC'] = df['SCC'].apply(map_scc)
    df['CALC'] = df['CALC'].apply(map_calc)
    df['MTRANS'] = df['MTRANS'].apply(map_transportation)

    # Scale df
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    prediction = model.predict(df)

    
    prediction = map_pred(prediction[0])

    print(df)
    print(prediction)

    prediction_label.config(text=str(prediction))
    

button = tk.Button(window, text="Submit", command=predict)
button.pack()

prediction_label = tk.Label(window, text="")
prediction_label.pack()

window.mainloop()