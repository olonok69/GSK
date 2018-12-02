from pyspark.ml.classification import  LogisticRegression, NaiveBayes
from pyspark.ml.classification import LogisticRegressionModel, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import shutil
import os

import logging
logger = logging.getLogger(__name__)

class c_estimator(object):

    def __train_model(self, alg, train_set):
        if alg == 1:
            logger.info("Training Logistic Regression...")
            lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
            model = lr.fit(train_set)
        elif alg ==2:
            logger.info("Training Naive Bayes...")
            nb = NaiveBayes(smoothing=1)
            model = nb.fit(train_set)
        return model

    def __test_model(self,test_set, model):
        # evaluate a model
        #
        logger.info("Testing  Logistic Regression...")
        predictions = model.transform(test_set)
        return predictions
    
    def __save_model(self,alg,model):
        if alg==1:
            dirpath ="lg_mode_path"
        elif alg==2:
            dirpath ="nb_mode_path"            
        
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        if alg==1:
            logger.info("Saving to disk Logistic Regression...")
            self.trained_model.save(dirpath)
        elif alg==2:
            logger.info("Saving to disk Naive Bayes...")                
            self.trained_model.save(dirpath)
        return
            
    def __load_model(self,alg):
        if alg==1:
            dirpath ="lg_mode_path"
        elif alg==2:
            dirpath ="nb_mode_path"            
        
        if alg==1:
            logger.info("loading from disk Logistic Regression Model...")
            self.predictions=LogisticRegressionModel.load(dirpath)
        elif alg==2:
            logger.info("loadind from disk Naive Bayes model...")                
            self.predictions=NaiveBayesModel.load(dirpath)
        return 

    def __evaluate_model(self, alg, predictions):
        if alg == 1:
            logger.info("Evaluating the model Logistic Regression...")
            evaluator = MulticlassClassificationEvaluator\
                (predictionCol="prediction", metricName="accuracy")
            Accuraccy=evaluator.evaluate(predictions)
        elif alg ==2:
            logger.info("Evaluating the model Naive Baves...")
            evaluator = MulticlassClassificationEvaluator\
                (predictionCol="prediction", metricName="accuracy")
            Accuraccy=evaluator.evaluate(predictions)
        print("\n")
        print(Accuraccy)
        print("\n")
        return Accuraccy


    def __init__(self,alg,train_set,test_set):
        self.alg=alg
        self.trained_model=self.__train_model(alg,train_set)
        self.predictions=self.__test_model(test_set,self.trained_model)
        self.saved_model=self.__save_model(alg,self.trained_model)
        self.accuracy=self.__evaluate_model(alg,self.predictions)
        			