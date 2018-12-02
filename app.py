from flask import Blueprint,Flask, request
main = Blueprint('main', __name__)
 
import json
from classification import ClassificationEngine

 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
#from flask import 
 
@main.route("/<int:alg>/prediction/accuracy/", methods=["GET"])
def get_accuracy(alg):
    logger.debug("Accuracy of Estimator:", alg)
    Accuraccy = Classification_engine.accuraccy_alg(alg)
    return json.dumps(Accuraccy)

@main.route("/<int:alg>/predictions", methods=["GET"])
def predict_new(alg):
    main_text= request.args.get('main_text', None) 
    add_text= request.args.get('add_text', None) 
    manufacturer= request.args.get('manufacturer', None) 
    #predict classe after send main text, additional text and manufacturer
    predicted_class=Classification_engine.predict_new\
    (alg,main_text,add_text,manufacturer)
    return predicted_class
    
  
 
def create_app(spark_context, dataset_path):
    global Classification_engine
    Classification_engine = ClassificationEngine(spark_context,\
	                   dataset_path,1)


    #train_data=Classification_engine.trainData
    #estimator=create_estimator(1,train_data)
    #model=estimator.trained_model
    #test_data=Classification_engine.TestData
    #predictions=test_model(test_data,model,estimator)
    
    
    app = Flask(__name__)
    app.register_blueprint(main)
    return app
    
"""    
def train_alg(est, alg, dataset):
    logger.info("Training the estimator...")
    model=est.__train_model(alg,dataset)
    return model

def test_model(estimator,TestData,model):
    logger.info("Testing the estimator...")

    return estimator
    
def create_estimator(alg,dataset):
    logger.info("Creating the estimator you choose...")
    return c_estimator(alg,dataset)
"""