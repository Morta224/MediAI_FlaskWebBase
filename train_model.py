import mysql.connector
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier

def connect_to_database():
    """Connect to MySQL database"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="medical_db"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def clean_symptoms(symptom_str):
    """Standardize symptom formatting while preserving all original symptoms"""
    symptom_str = symptom_str.lower().strip()
    
    # Create a mapping of variations to standard forms without replacing
    symptom_mapping = {
        'ache': ('pain', 'ache'),
        'swell': ('swelling', 'swell'),
        'hurts': ('pain', 'hurts'),
        'sore': ('pain', 'sore'),
        'hurting': ('pain', 'hurting'),
        'tenderness': ('pain', 'tenderness'),
        'discomfort': ('pain', 'discomfort'),
        'throbbing': ('pain', 'throbbing'),
        'stiffness': ('pain', 'stiffness'),
        'cramp': ('pain', 'cramp'),
        'nausea': ('nausea_and_vomiting', 'nausea'),
        'vomiting': ('nausea_and_vomiting', 'vomiting'),
        'feverish': ('fever', 'feverish'),
        'temp': ('fever', 'temp'),
        'urinate': ('urination', 'urinate'),
        'pee': ('urination', 'pee'),
        'bm': ('bowel_movement', 'bm'),
        'poop': ('bowel_movement', 'poop'),
    }
    
    # Return both the standardized form and original symptom
    if symptom_str in symptom_mapping:
        return symptom_mapping[symptom_str]
    return (symptom_str, symptom_str)

def load_and_preprocess_data(conn):
    """Load and preprocess data while preserving all symptom variations"""
    cursor = conn.cursor()
    cursor.execute("SELECT symptoms, disease, treatments FROM symptoms_data")
    rows = cursor.fetchall()
    
    X = []
    y = []
    symptom_counter = Counter()
    disease_counter = Counter()
    treatments_dict = {}
    symptom_variations = {}  # Track all variations for each symptom

    for symptoms, disease, treatments in rows:
        processed_symptoms = []
        for s in symptoms.split(','):
            if s.strip():
                standardized, original = clean_symptoms(s)
                processed_symptoms.append(standardized)
                symptom_counter[standardized] += 1
                
                # Track all variations for this standardized symptom
                if standardized not in symptom_variations:
                    symptom_variations[standardized] = set()
                symptom_variations[standardized].add(original)
        
        disease_cleaned = disease.strip()
        X.append(processed_symptoms)
        y.append(disease_cleaned)
        disease_counter[disease_cleaned] += 1

        treatments_dict[disease_cleaned] = treatments.strip() if treatments else "No treatments data"
    
    print("\nTop 20 Symptoms (Standardized Forms):")
    print(symptom_counter.most_common(20))
    
    print("\nSymptom Variations:")
    for std_symptom, variations in symptom_variations.items():
        print(f"{std_symptom}: {', '.join(variations)}")
    
    print("\nDisease Distribution:")
    print(disease_counter.most_common(20))
    
    return X, y, symptom_counter, treatments_dict, symptom_variations

def train_and_evaluate(X, y):
    """Train and evaluate model"""
    mlb = MultiLabelBinarizer()
    X_encoded = mlb.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Calculate class weights
    weights = class_weight.compute_sample_weight('balanced', y_train)
    
    # Train models
    print("\nTraining MultinomialNB...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train, sample_weight=weights)
    
    print("\nTraining RandomForest...")
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    print("\nNaive Bayes Evaluation:")
    print(classification_report(y_test, nb_model.predict(X_test)))
    
    print("\nRandom Forest Evaluation:")
    print(classification_report(y_test, rf_model.predict(X_test)))
    
    return rf_model, mlb  # Using RandomForest as it typically performs better

def save_model(model, mlb, symptom_counter, treatments_dict, symptom_variations):
    """Save model and artifacts including symptom variations"""
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'mlb': mlb,
            'common_symptoms': [s[0] for s in symptom_counter.most_common(100)],
            'symptom_mapping': {k: clean_symptoms(k)[0] for k in symptom_counter},
            'symptom_variations': symptom_variations,
            'treatments_dict': treatments_dict
        }, f)
    print("\nModel saved successfully with symptom variations")

def main():
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        X, y, symptom_counter, treatments_dict, symptom_variations = load_and_preprocess_data(conn)
        model, mlb = train_and_evaluate(X, y)
        save_model(model, mlb, symptom_counter, treatments_dict, symptom_variations)
    finally:
        conn.close()

if __name__ == "__main__":
    main()