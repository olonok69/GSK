from pyspark.ml.classification import  LogisticRegression, NaiveBayes
from pyspark.ml.classification import LogisticRegressionModel, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import shutil
import os

import logging
logger = logging.getLogger(__name__)

class c_estimator(object):
	
	def __train_model(self, alg, train_set):
	
		if alg== 1:
			lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
			model = lr.fit(train_set)
		elif alg==2:
			nb = NaiveBayes(smoothing=1)
			model = nb.fit(train_set)
		return model
		
	def __test_model(self,test_set, model):
		# evaluate a model
		#
		predictions = model.transform(test_set)
		
		return predictions
		
	def	__save_model(self,alg,path,model):
		dirpath = os.path.join(path)
		if os.path.exists(dirpath) and os.path.isdir(dirpath):
				shutil.rmtree(dirpath)
		if alg==1:
			model.save("lg_mode_path")
		elif alg==2:
			model.save("nb_mode_path")
		return

	def __init__(self,session, alg):
		self.session=session
		self.alg=alg			