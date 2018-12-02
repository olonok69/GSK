import shutil
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
    
    def __transform_data(self,dataset,prediction):
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
        
        manufacturer_Y = StringIndexer(inputCol = "manufacturer", \
            outputCol = "manufacturer_C")
        
        # OneHotEncoder to Manufacturers to help Classifier    
        # encoders2 = OneHotEncoder(inputCol=manufacturer_Y.getOutputCol(), \
        # outputCol="manufacture_1H")     
        #create First Pipeline
        #if prediction==1:
        data_prep_pipe = Pipeline(stages=[regexTokenizer,stopwordsRemover,\
                            hashingTF,idf,regexTokenizer2,stopwordsRemover2,\
                            hashingTF2,idf2])
        # elif prediction==2:
        #     data_prep_pipe = Pipeline(stages=[regexTokenizer,stopwordsRemover,\
        #                     hashingTF,idf,regexTokenizer2,stopwordsRemover2,\
        #                     hashingTF2,idf2])
        #                     
        data_transformer = data_prep_pipe.fit(dataset)
        data = data_transformer.transform(dataset)
        
        data=manufacturer_Y.fit(data).transform(data)

        # String Category To numbers
        if prediction==1:
            product_group_Y = StringIndexer(inputCol = "product_group", \
                outputCol = "label")
            # Second Pipeline
            data=product_group_Y.fit(data).transform(data)

        return data   
        
 
    def __Vector_Assembler(self,dataset,prediction):
        #crete Feature Vectors to train estimators
        clean_up = VectorAssembler(inputCols=['rawFeatures1','rawFeatures2',\
            'manufacturer_C'],outputCol='features')  
        dataout=clean_up.transform(dataset)
        # dirpath="VectorAssembler"
        # if os.path.exists(dirpath) and os.path.isdir(dirpath):
        #     shutil.rmtree(dirpath)
        # self.Vector=clean_up.save("VectorAssembler")
        if prediction==1:
            dataout=dataout.select("main_text","add_text","product_group",\
                "features",'label')
        return dataout
            
    def data_split(self,dataset):
        # set seed for reproducibility
        (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
        return trainingData, testData
    
    def create_grouped_object(self, dataset):
        datagroup=dataset.groupby('product_group').avg('label')
        datagroup=datagroup.withColumnRenamed("avg(label)", "label")
        return datagroup
        
    def accuraccy_alg(self, alg):
        if alg==1:
            accuraccy="Accuraccy Logistic Regression estimator: {}".format\
            (self.lg_estimator.accuracy)           
        elif alg==2:
            accuraccy="Naives Bayes estimator: {}".format\
            (self.lg_naives.accuracy)           
        return accuraccy

    def predict_new(self,alg, main_text,add_text,manufacturer):
        dataframe=self.create_dataframe(main_text,add_text,manufacturer)
        dataframe_tr=self.__transform_data(dataframe,2)
        dataout_new=self.__Vector_Assembler(dataframe_tr,2)
        #dataout_new=vassembler.transform(dataframe_tr)
        dataout_new=dataout_new.filter(sf.col('add_text')!="fake")
        
        print(dataout_new.printSchema())
        if alg==1:
            prediction=self.lg_estimator.trained_model.transform(dataout_new)
        elif alg==2:
            prediction=self.lg_naives.trained_model.transform(dataout_new)
        
        temp1=prediction.join(self.datagroup, prediction.prediction==\
        self.datagroup.label ,how='left').select(prediction.product_group,\
        prediction.probability, self.datagroup.product_group)
        out=temp1.collect()
        probability=max(out[0]['probability'])
        category=out[0][2]
        if alg==1:
            algo="Logistic Regression"
        elif  alg==2:
            algo= "Naive Bayes"
        string_cat="The category predicted is {} with a probability of {}.\
         The algorithm use for preditions is {}. ".\
        format(category,probability,algo)
        
        return string_cat
  
    def create_dataframe(self,main_text,add_text,manufacturer):
        columns=['product_group','main_text','add_text','manufacturer']
        vals=[("NO_GROUP",str(main_text),str(add_text),str(manufacturer)),
        ("NO_GROUP",str(main_text),"fake","NO_Manufacturer")]
        df=self.sc.createDataFrame(vals, columns)
        return df



    def __init__(self, session, dataset_path,alg):

        self.Vector=None
        logger.info("Starting up the Classification Engine: ")
        self.sc = session
        # Load data for training ALG,s
        logger.info("Loading Data Training...")
        self.dataset=self.__load_training_file("Dataset_N.csv", self.sc)
        
        logger.info("Preprocessing data...")
        self.dataset_pre=self.__data_preprocessing(self.dataset)
     
        logger.info("First Transformation data...")
        self.dataset_tr=self.__transform_data(self.dataset_pre,1)

        logger.info("Creating grouped objects...")
        self.datagroup=self.create_grouped_object(self.dataset_tr)
                
        logger.info("Second Transformation data. Feature Creation...")
        self.dataset_tr=self.__Vector_Assembler(self.dataset_tr,1)
        
        logger.info("Splitting the data...")
        self.trainData, self.TestData=self.data_split(self.dataset_tr)
        
        print("Training Dataset Count: " + str(self.trainData.count()))
        print("Test Dataset Count: " + str(self.TestData.count()))

        logger.info("Creating Estimators...")
        logger.info("Creating Logistic Regression...")
        self.lg_estimator=c_estimator(1,self.trainData,self.TestData)
        
        logger.info("Creating Naives Bayes...")
        self.lg_naives=c_estimator(2,self.trainData,self.TestData)
        
        
        
