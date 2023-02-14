import os
import sys

from predict.exception import PredictException
from predict.util import load_object

import pandas as pd

class CreditRisk:
    def __init__(self,
        status,
        duration,
        credit_history,
        purpose,
        amount,
        savings,
        employment_duration,
        installment_rate,
        personal_status_sex,
        other_debtors,
        present_residence,
        prop,
        age,
        other_installment_plans,
        housing,
        number_credits,
        job,
        people_liable,
        telephone,
        foreign_worker
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
                "status":[self.status],
                "duration":[self.duration],
                "credit_history":[self.credit_history],
                "purpose":[self.purpose],
                "amount":[self.amount],
                "savings":[self.savings],
                "employment_duration":[self.employment_duration],
                "installment_rate":[self.installment_rate],
                "personal_status_sex":[self.personal_status_sex],
                "other_debtors":[self.other_debtors],
                "present_residence":[self.present_residence],
                "prop":[self.prop],
                "age":[self.age],
                "other_installment_plans":[self.other_installment_plans],
                "housing":[self.housing],
                "number_credits":[self.number_credits],
                "job":[self.job],
                "people_liable":[self.people_liable],
                "telephone":[self.telephone],
                "foreign_worker":[self.foreign_worker]
                }
            return input_data
        except Exception as e:
            raise PredictException(e, sys)
        
        
class CreditRiskPreditor:
    def __init__(self, model_dir:str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise PredictException(e, sys) from e
        
    def get_latest_model_path(self):
        try:
            folder_name = list(map(int,os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise PredictException(e, sys) from e
        
    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(model_path)
            estimated_credit_risk = model.predict(X)
            return estimated_credit_risk
        except Exception as e:
            raise PredictException(e, sys) from e
        
        