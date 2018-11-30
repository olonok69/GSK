import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover,RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from estimators import c_estimator
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationEngine(object):
 
    def __load_training_file(self,dataset_path,session):
        data=session.read.csv("Dataset_N.csv", inferSchema=True,sep=';'\
                            ,header=True)
        return data

    def __data_preprocessing(self,dataset):

        d=dataset.withColumnRenamed("V1","ID").\
        withColumnRenamed("V2","product_group").\
        withColumnRenamed("V3","main_text").\
        withColumnRenamed("V4","add_text").\
        withColumnRenamed("V5","manufacturer")
        
        d=d.select('product_group','main_text','add_text','manufacturer')
        
        # remove NULL because cause a lot of issues
        d=d.fillna({'manufacturer':"NO_Manufacturer"})
        d=d.fillna({'main_text':"NO_TEXT"})
        return d
    
    def __transform_data(self,dataset):
        # regular expression tokenizer + StopWordsRemover + Frequency Term +
        # Inverse Document Frequency
        regexTokenizer = RegexTokenizer(inputCol="main_text", \
        outputCol="main_text_t")
        stopwordsRemover = StopWordsRemover(inputCol="main_text_t", \
        outputCol="main_text_f")
        hashingTF = HashingTF(inputCol="main_text_f", outputCol="rawFeatures1")
        idf = IDF(inputCol="rawFeatures1", outputCol="tf_idf")

        # regular expression tokenizer 2 block
        regexTokenizer2 = RegexTokenizer(inputCol="add_text",\
        outputCol="add_text_t")
        stopwordsRemover2 = StopWordsRemover(inputCol="add_text_t", \
        outputCol="add_text_f")
        hashingTF2 = HashingTF(inputCol="add_text_f", outputCol="rawFeatures2")
        idf2 = IDF(inputCol="rawFeatures2", outputCol="tf_idf2")
        #create First Pipeline
        data_prep_pipe = Pipeline(stages=[regexTokenizer,stopwordsRemover,\
                            hashingTF,idf,regexTokenizer2,stopwordsRemover2,\
                            hashingTF2,idf2])
                            
        data_transformer = data_prep_pipe.fit(dataset)
        data = data_transformer.transform(dataset)

        # String Category To numbers
        product_group_Y = StringIndexer(inputCol = "product_group", \
                outputCol = "label")
        manufacturer_Y = StringIndexer(inputCol = "manufacturer", \
            outputCol = "manufacturer_C")

        # OneHotEncoder to Manufacturers to help Classifier    
        encoders2 = OneHotEncoder(inputCol=manufacturer_Y.getOutputCol(), \
        outputCol="manufacture_1H")     
        # Second Pipeline
        pipe2=Pipeline(stages=[product_group_Y,manufacturer_Y,encoders2])
        data=pipe2.fit(data).transform(data)
        return data   
        
 
    def __Vector_Assembler(self,dataset):
        #crete Feature Vectors to train estimators
        clean_up = VectorAssembler(inputCols=['rawFeatures1','rawFeatures2',\
            'manufacture_1H'],outputCol='features')  
        dataout=clean_up.transform(dataset)
        datax=dataout.select("main_text","add_text","product_group",\
                "features",'label')
        return datax
            
    def __data_split(self,dataset):
        # set seed for reproducibility
        (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
        return trainingData, testData

    def train_alg(self, est, alg):
        logger.info("Training the estimator...")
        model=est.__train_model(alg, self.trainData)
        return self.model

    def __test_alg(self, est,TestData,model):
        predictions=est.__test_model(TestData, model)
        return predictions

    def __create_estimator(self, session,alg):
        logger.info("Creating the estimator you choose...")
        self.estimator=c_estimator(session, alg)
        return self.estimator

    def __init__(self, session, dataset_path,alg):


        logger.info("Starting up the Classification Engine: ")
        self.sc = session
        # Load data for training ALG,s
        logger.info("Loading Data Training...")
        self.dataset=self.__load_training_file("Dataset_N.csv", self.sc)
        logger.info("Preprocessing data...")
        self.dataset_pre=self.__data_preprocessing(self.dataset)
        logger.info("First Transformation data...")
        self.dataset_tr=self.__transform_data(self.dataset_pre)
        logger.info("Second Transformation data. Feature Creation...")
        self.dataset_tr=self.__Vector_Assembler(self.dataset_tr)
        logger.info("Splitting the data...")
        self.trainData, self.TestData=self.__data_split(self.dataset_tr)
        #print("Training Dataset Count: " + str(self.trainData.count()))
        #print("Test Dataset Count: " + str(self.TestData.count()))
        logger.info("Creating the estimator you choose...")
        self.estimator=self.__create_estimator(session, alg )
        #self.est=c_estimator(session, 1)
        #self.model=self.__train_alg(self.est, self.trainData,1)
        #self.predictions=self.__test_alg(self.est,self.TestData,self.model)
        #predictions=self.predictions

        #print(predictions.filter(predictions['prediction'] == 0) \
        #	.select("main_text","add_text","product_group","probability","label","prediction") \
        #		.orderBy("probability", ascending=False) \
        #	.show(n = 10, truncate = 30))