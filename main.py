from flask import Flask, request, render_template, jsonify, flash
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path


app = Flask(__name__)
app.secret_key = os.urandom(24) 


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"


os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


try:
    sym_des = pd.read_csv(DATASET_DIR / "symtoms_df.csv")
    precautions = pd.read_csv(DATASET_DIR / "precautions_df.csv")
    workout = pd.read_csv(DATASET_DIR / "workout_df.csv")
    description = pd.read_csv(DATASET_DIR / "description.csv")
    medications = pd.read_csv(DATASET_DIR / "medications.csv")
    diets = pd.read_csv(DATASET_DIR / "diets.csv")

   
    with open(MODEL_DIR / "svc.pkl", 'rb') as model_file:
        svc = pickle.load(model_file)
except Exception as e:
    print(f"Error loading files: {e}")
    raise

def helper(dis):
    """Get disease information"""
    try:
       
        desc = description[description['Disease'] == dis]['Description']
        desc = " ".join([str(w) for w in desc]) if not desc.empty else "No description available"

       
        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [str(col) for col in pre.values[0]] if not pre.empty else ["No precautions available"]

        
        med = medications[medications['Disease'] == dis]['Medication']
        med = [str(m) for m in med.values] if not med.empty else ["No medications listed"]

        
        die = diets[diets['Disease'] == dis]['Diet']
        die = [str(d) for d in die.values] if not die.empty else ["No diet recommendations available"]

       
        wrkout = workout[workout['disease'] == dis]['workout']
        wrkout = [str(w) for w in wrkout.values] if not wrkout.empty else ["No workout recommendations available"]

        return desc, pre, med, die, wrkout
    except Exception as e:
        print(f"Error in helper function: {e}")
        return "Error occurred", [], [], [], []


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptom_mapping = {
   
    'fever': 'mild_fever',
    'high temperature': 'high_fever',
    'temperature': 'mild_fever',
    
   
    'weakness': 'fatigue',
    'tired': 'fatigue',
    'sleeping': 'fatigue',
    'exhaustion': 'fatigue',
    'feeling weak': 'fatigue',
    
  
    'stomach ache': 'stomach_pain',
    'belly ache': 'belly_pain',
    'head ache': 'headache',
    'joint aches': 'joint_pain',
    'muscle aches': 'muscle_pain',
    
  
    'throwing up': 'vomiting',
    'dizzy': 'dizziness',
    'feeling sick': 'nausea',
    'cant eat': 'loss_of_appetite',
    'no appetite': 'loss_of_appetite',
    'sweaty': 'sweating',
    'thirsty': 'dehydration',
    'cant breathe': 'breathlessness',
    'breathing problem': 'breathlessness',
    'hard to breathe': 'breathlessness'
}

def preprocess_symptoms(symptoms):
    """Preprocess symptoms to handle common variations"""
    processed = []
    for symptom in symptoms:
        
        symptom = symptom.lower().strip()
        
       
        if symptom in symptom_mapping:
            processed.append(symptom_mapping[symptom])
        else:
            
            processed.append(symptom)
    return processed

feature_names = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]


assert len(feature_names) == len(symptoms_dict), "Feature names length doesn't match symptoms dictionary"
for i, feature in enumerate(feature_names):
    assert symptoms_dict[feature] == i, f"Feature {feature} is not in the correct position"

def get_predicted_value(patient_symptoms):
    """Predict disease based on symptoms"""
    try:
      
        valid_symptoms = [s.strip().lower() for s in patient_symptoms if s.strip().lower() in symptoms_dict]
        if not valid_symptoms:
            return None

        input_vector = [0] * len(feature_names)
        
      
        for symptom in valid_symptoms:
            input_vector[symptoms_dict[symptom]] = 1

       
        input_df = pd.DataFrame([input_vector], columns=feature_names)
        
        try:
            prediction = svc.predict(input_df)[0]
            predicted_disease = diseases_list.get(prediction, "Unknown condition")
            
           
            if len(valid_symptoms) == 1:
               
                if valid_symptoms[0] in ['cough', 'mild_fever', 'high_fever']:
                    return 'Common Cold'
                elif valid_symptoms[0] in ['fatigue', 'weakness_in_limbs']:
                    return 'Viral infection'
            elif len(valid_symptoms) >= 2:
               
                if 'cough' in valid_symptoms and ('mild_fever' in valid_symptoms or 'high_fever' in valid_symptoms):
                    if 'breathlessness' in valid_symptoms or 'chest_pain' in valid_symptoms:
                        return 'Pneumonia'
                    else:
                        return 'Common Cold'
                        
                if 'fatigue' in valid_symptoms and 'muscle_pain' in valid_symptoms:
                    if 'high_fever' in valid_symptoms:
                        return 'Dengue'
                    else:
                        return 'Viral infection'
            
            return predicted_disease
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            
            if len(valid_symptoms) == 1:
                if valid_symptoms[0] in ['cough', 'mild_fever', 'high_fever']:
                    return 'Common Cold'
                elif valid_symptoms[0] in ['fatigue', 'weakness_in_limbs']:
                    return 'Viral infection'
            return "Unknown condition"
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route("/")
def index():
    """Home page route"""
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction route"""
    if request.method == 'POST':
        try:
         
            symptoms = request.form.get('symptoms', '').strip()
            print(f"Received symptoms: {symptoms}")
            
            if not symptoms or symptoms.lower() == "symptoms":
                flash("Please enter valid symptoms")
                return render_template('index.html')

            
            user_symptoms = [s.strip().lower() for s in symptoms.split(',') if s.strip()]
            print(f"Processed symptoms: {user_symptoms}")
            
        
            user_symptoms = preprocess_symptoms(user_symptoms)
            print(f"Preprocessed symptoms: {user_symptoms}")
            
          
            valid_symptoms = [s for s in user_symptoms if s in symptoms_dict]
            invalid_symptoms = [s for s in user_symptoms if s not in symptoms_dict]
            
            if invalid_symptoms:
                flash(f"Unrecognized symptoms: {', '.join(invalid_symptoms)}")
                flash("Try using more specific terms or check the common symptoms list below:")
                flash("Common symptoms: cough, fever, fatigue, headache, joint_pain, muscle_pain, etc.")
                return render_template('index.html')
            
            if not valid_symptoms:
                flash("Please enter valid symptoms from the available list")
                flash("Common symptoms: cough, fever, fatigue, headache, joint_pain, muscle_pain, etc.")
                return render_template('index.html')

            print(f"Valid symptoms for prediction: {valid_symptoms}")
            
           
            predicted_disease = get_predicted_value(valid_symptoms)
            print(f"Predicted disease: {predicted_disease}")
            
            if not predicted_disease:
                flash("Could not make a prediction. Please check your symptoms and try again.")
                return render_template('index.html')

           
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            print(f"Retrieved information for: {predicted_disease}")

           
            flash(f"Prediction based on symptoms: {', '.join(valid_symptoms)}")
            if len(valid_symptoms) < 3:
                flash("For more accurate predictions, please provide at least 3 symptoms")

            return render_template('index.html',
                                predicted_disease=predicted_disease,
                                dis_des=dis_des,
                                my_precautions=precautions,
                                medications=medications,
                                my_diet=rec_diet,
                                workout=workout)

        except Exception as e:
            print(f"Error in prediction route: {str(e)}")  
            flash(f"An error occurred: {str(e)}")
            return render_template('index.html')

    return render_template('index.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template("about.html")

@app.route('/contact')
def contact():
    """Contact page route"""
    return render_template("contact.html")

@app.route('/blog')
def blog():
    """Blog page route"""
    return render_template("blog.html")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)