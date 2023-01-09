import os, sys

class PredictException(Exception):
    def __init__(self,error_message:Exception, error_details:sys) :
        super.__init__(error_message)
        self.error_message = PredictException.get_error_message(error_message, error_details)
        
    @staticmethod
    def get_error_message(self,error_message:Exception, error_details:sys):
        _,_,exct_tb = error_details.exc_info()
        line_number = exct_tb.tb_lineno
        file_name = exct_tb.tb_frame.f_code.co_name
        return f"""Error Occured at
                                Line Number: {line_number} in 
                                File Name: {file_name} is 
                                Error message {error_message}"""
                                
    def __str__(self) :
        return self.error_message
    
    def __repr__(self) :
        return PredictException.__name__.str()

