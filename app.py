from flask import Flask, render_template, request
import pickle
import numpy as np
import logging
from collections import Counter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load Model & Artifacts ---
def load_model(): 
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            mlb = model_data['mlb']
            common_symptoms = model_data['common_symptoms']
            symptom_mapping = model_data['symptom_mapping']
            treatments_dict = model_data['treatments_dict']
        
        logging.info(f"Model loaded. Diseases: {len(model.classes_)}, Symptoms: {len(mlb.classes_)}")
        return model, mlb, common_symptoms, symptom_mapping, treatments_dict
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

model, mlb, common_symptoms, symptom_mapping, treatments_dict = load_model()

# --- Helper Functions ---
def preprocess_input(symptoms_str):
    """Clean and map input symptoms."""
    symptoms = [s.strip().lower() for s in symptoms_str.split(',') if s.strip()]
    return [symptom_mapping.get(s, s) for s in symptoms]

def predict_disease(symptoms, min_prob=0.05, top_n=5):
    """Predict diseases with given symptoms."""
    encoded = mlb.transform([symptoms])
    proba = model.predict_proba(encoded)[0]
    
    predictions = [
        (disease, f"{prob*100:.1f}%", treatments_dict.get(disease, "No treatments available"))
        for disease, prob in zip(model.classes_, proba)
        if prob > min_prob
    ]
    predictions.sort(key=lambda x: float(x[1][:-1]), reverse=True)
    return predictions[:top_n]

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    raw_input = ""
    warning = None
    suggestions = []

    if request.method == 'POST':
        raw_input = request.form['symptoms_input'].strip()
        
        if not raw_input or all(c in ', ' for c in raw_input):
            predictions = [("No valid symptoms entered", "Please enter at least 1 symptom", "")]
        else:
            try:
                input_symptoms = preprocess_input(raw_input)
                
                # Check for unrecognized symptoms
                uncommon = [s for s in input_symptoms if s not in mlb.classes_]
                if uncommon:
                    warning = f"Note: These symptoms may not be recognized: {', '.join(uncommon)}"
                    suggestions = [s for s in common_symptoms if any(u in s for u in uncommon)][:3]
                
                predictions = predict_disease(input_symptoms)
                
                if not predictions:
                    predictions = [("No confident prediction found", "Try more specific symptoms", "")]
                    
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                predictions = [("Error processing symptoms", "Please check your input and try again", "")]

    return render_template(
        'index.html',
        predictions=predictions,
        user_input=raw_input,
        warning=warning,
        suggestions=suggestions,
        example_symptoms=", ".join(common_symptoms[:5])
    )

if __name__ == '__main__':
    app.run(debug=True)