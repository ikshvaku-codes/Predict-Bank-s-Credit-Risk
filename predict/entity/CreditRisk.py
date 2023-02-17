import os
import sys

from predict.exception import PredictException
from predict.util import load_object
from predict.component.ModelTrainer import predictEstimatorModel
import pandas as pd

class CreditRisk:
    def __init__(self,
        status:float,
        duration:float,
        credit_history:float,
        purpose:float,
        amount:float,
        savings:float,
        employment_duration:float,
        installment_rate:float,
        personal_status_sex:float,
        other_debtors:float,
        present_residence:float,
        prop:float,
        age:float,
        other_installment_plans:float,
        housing:float,
        number_credits:float,
        job:float,
        people_liable:float,
        telephone:float,
        foreign_worker:float
    ):
        try:
            self.status = status 
            self.duration = duration 
            self.credit_history = credit_history 
            self.purpose = purpose
            self.amount = amount 
            self.savings = savings
            self.employment_duration = employment_duration 
            self.installment_rate = installment_rate 
            self.personal_status_sex = personal_status_sex 
            self.other_debtors = other_debtors
            self.present_residence = present_residence 
            self.prop = prop
            self.age = age 
            self.other_installment_plans = other_installment_plans 
            self.housing = housing 
            self.number_credits = number_credits 
            self.job = job 
            self.people_liable = people_liable 
            self.telephone = telephone 
            self.foreign_worker = foreign_worker 
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def get_credit_risk_input_data_frame(self):
        try:
            credit_risk_dict = self.get_credit_risk_data_as_dict()
            return pd.DataFrame(credit_risk_dict)
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def get_credit_risk_data_as_dict(self):
        try:
            input_data = {
                "laufkont":[self.status],
                "laufzeit":[self.duration],
                "moral":[self.credit_history],
                "verw":[self.purpose],
                "hoehe":[self.amount],
                "sparkont":[self.savings],
                "beszeit":[self.employment_duration],
                "rate":[self.installment_rate],
                "famges":[self.personal_status_sex],
                "buerge":[self.other_debtors],
                "wohnzeit":[self.present_residence],
                "verm":[self.prop],
                "alter":[self.age],
                "weitkred":[self.other_installment_plans],
                "wohn":[self.housing],
                "bishkred":[self.number_credits],
                "beruf":[self.job],
                "pers":[self.people_liable],
                "telef":[self.telephone],
                "gastarb":[self.foreign_worker]
                }
            return input_data
        except Exception as e:
            raise PredictException(e, sys)
        
        
class CreditRiskPreditor:
    def __init__(self, model_dir:str, preprocessing_dir:str):
        try:
            self.model_dir = model_dir
            self.preprocessing_dir = preprocessing_dir
        except Exception as e:
            raise PredictException(e, sys) from e
        
    def get_latest_model_path(self):
        try:
            path_list = os.listdir(self.model_dir)
            folder_name = {
                int(value.replace("-","")): value for value in path_list
            }
            
            latest_model_dir = os.path.join(self.model_dir, f"{folder_name[max(folder_name.keys())]}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise PredictException(e, sys) from e
        
    def get_latest_preprocessing_obj_path(self):
        try:
            path_list = os.listdir(self.preprocessing_dir)
            folder_name = {
                int(value.replace("-","")): value for value in path_list
            }
            
            latest_preprocessing_dir = os.path.join(self.preprocessing_dir, f"{folder_name[max(folder_name.keys())]}", "preprocessed")
            file_name = os.listdir(latest_preprocessing_dir)[0]
            latest_preprocessing_dir = os.path.join(latest_preprocessing_dir, file_name)
            return latest_preprocessing_dir
        except Exception as e:
            raise PredictException(e, sys) from e
        
    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            preprocessing_obj_path = self.get_latest_preprocessing_obj_path()
            
            model = load_object(model_path)
            preprocessing_obj = load_object(preprocessing_obj_path)
            
            # estimated_credit_risk_obj = predictEstimatorModel(preprocessing_obj, model)
            # estimated_credit_risk = estimated_credit_risk_obj.predict(X)
            estimated_credit_risk = model.predict(X)
            return estimated_credit_risk
        except Exception as e:
            raise PredictException(e, sys) from e
        
        