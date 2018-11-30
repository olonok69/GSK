from flask import Blueprint
main = Blueprint('main', __name__)
 
import json
from classification import ClassificationEngine
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
from flask import Flask, request
 

 
 
def create_app(spark_context, dataset_path):
    global Classification_engine 
 
    Classification_engine = ClassificationEngine(spark_context, dataset_path)    
    
    app = Flask(__name__)
    app.register_blueprint(main)
    return app