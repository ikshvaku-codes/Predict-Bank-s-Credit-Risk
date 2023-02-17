from flask import Flask, request
from predict.pipeline.pipeline import Pipeline
from flask import send_file, abort, render_template
import os
from predict.constant import *
import pandas as pd
from predict.entity.CreditRisk import CreditRisk, CreditRiskPreditor

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "predict_logs"
PIPELINE_FOLDER_NAME = "predict"
ARTIFACT_FOLDER_NAME = "artifact"
SAVED_MODELS_DIR_NAME = "saved_models"
PREPROCESSING_OBJ_DIR_NAME = "data_transformation"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config", "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(PIPELINE_DIR, ARTIFACT_FOLDER_NAME, SAVED_MODELS_DIR_NAME)
PREPROCESSING_OBJ_DIR_PATH = os.path.join(PIPELINE_DIR,ARTIFACT_FOLDER_NAME, PREPROCESSING_OBJ_DIR_NAME)

CREDIT_DATA_KEY = "customers_data"
ESTIMATED_CREDIT_RISK_VALUE_KEY = "estimated_credit_risk"

app = Flask(__name__)
pipeline = Pipeline()

@app.route('/artifact', defaults={'req_path': 'predict'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("predict", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/view_experiment_hist', methods =['GET', 'POST'])
def view_experiment_hist():
    experiment_df = Pipeline.get_experiments_status()
    if experiment_df is None:
        dictt = dict()
        dictt = {"status":["Pipeline is not Initiated"]}
        experiment_df = pd.DataFrame.from_dict(dictt)
        context = {
        "experiment": experiment_df.to_html(classes = 'table table-striped col-12')
        }
    else:
        
        context = {
            "experiment": experiment_df.to_html(classes = 'table table-striped col-12')
        }
    return render_template('experiment_history.html', context = context)

@app.route("/train")
def train_model():
    # change model configurations in model_factory.py->get_sample_model_config_yaml_file
    pipeline.start()
    
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
            CREDIT_DATA_KEY: None,
            ESTIMATED_CREDIT_RISK_VALUE_KEY: None
        }
    if request.method == 'POST':
        
        status = request.form['status'] 
        duration = request.form['duration']
        credit_history = request.form['credit_history']
        purpose = request.form['purpose']
        amount = request.form['amount']
        savings = request.form['savings']
        employment_duration = request.form['employment_duration']
        installment_rate = request.form['installment_rate']
        personal_status_sex = request.form['personal_status_sex']
        other_debtors = request.form['other_debtors']
        present_residence = request.form['present_residence']
        prop = request.form['property']
        age = request.form['age']
        other_installment_plans = request.form['other_installment_plans']
        housing = request.form['housing']
        number_credits = request.form['number_credits']
        job = request.form['job']
        people_liable = request.form['people_liable']
        telephone = request.form['telephone']
        foreign_worker = request.form['foreign_worker']

        credit_risk = CreditRisk(
            float(status.strip()),
            float(duration.strip()),
            float(credit_history.strip()),
            float(purpose.strip()),
            float(amount.strip()),
            float(savings.strip()),
            float(employment_duration.strip()),
            float(installment_rate.strip()),
            float(personal_status_sex.strip()),
            float(other_debtors.strip()),
            float(present_residence.strip()),
            float(prop.strip()),
            float(age.strip()),
            float(other_installment_plans.strip()),
            float(housing.strip()),
            float(number_credits.strip()),
            float(job.strip()),
            float(people_liable.strip()),
            float(telephone.strip()),
            float(foreign_worker.strip())
        )

        credit_risk_df = credit_risk.get_credit_risk_input_data_frame()
        credit_risk_estimator =  CreditRiskPreditor(MODEL_DIR, PREPROCESSING_OBJ_DIR_PATH)
        estimated_credit_risk = credit_risk_estimator.predict(credit_risk_df)
        context = {
            CREDIT_DATA_KEY: credit_risk.get_credit_risk_data_as_dict(),
            ESTIMATED_CREDIT_RISK_VALUE_KEY: estimated_credit_risk
        }
    
    return render_template('predict.html', context=context)

if __name__ == "__main__":
    app.run(debug=True)


