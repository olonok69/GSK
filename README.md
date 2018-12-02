### HTTP REST-API for ML MultiClass Classification Problem 

In this document is described the solution proposed for this problem. I use pyspark 2.3.4 and
flask 1.0.2 , Cherrypy 17.4.1

from Spark ML library I use two estimators Logistic Regression and Naive Bayes with their Evaluation models.
The solution proposed is fully modular and allows integrate new estimators or functionality and just plug-in
into the http server. Entry points are defined on the REST interface and can be query via GET request. In this
version I only provide 2 basic funtionalities , get the accuracy of the estimators and prediction of product class
via GET request with 3 paramethers Main Text, Additional Text and provider. ID is not use as it doesnt add any value
to the predictors. 
For data preparation I use NLP tecniques and generate TF-IDF vectors for main text and add text. For product category
and provider I use StringIndexers to transform string categories to numerical categories. Additionally i transformed 
initially the providers numerical category to a numerical vector category with One-HoT-Encoders, but finally i discarded
this option as it cant be use in production for prediction of single entities. The problem is that One-HoT-Encoders create
vector with a length according to the number of categories we have in providers(In our case 720 aprox diferent providers).
this means that when we train the estimator, we provide a vector of 720 legnth as input for providers, and this length is
use for build the model. In production when you submit a request to predict , we send just 1 observation and the transformation
process will create a One-Hot-Vector of length 1. This cause that prediction cant work as the feature vector is provided 
have diferent length to the one use on training. 

The application have 4 files:

- server.py entry point to start the spark session and invoque the application running on the Cherry server.
- app.py HTTP REST interface with 2 entry points and connection with the main back-end Class
- classification.py. This class govern the process of data preprocessing and transformation, instantiate the estimators , 
	training ,evaluation and prediction
- estimators.py. this class create estimator objects. In this version i only provide 2, but easily is possible with some 
	lines of code introduce any estimator available on spark MLlib, scikit-learnt or Neural Networks
	
One the application is submitted to the cluster, enters on data preprocessing and transform phase, instantiate the 2 
estimators and traing them. One their trainned  the model is serialized to disk, to inmediately load the model and use it 
for evaluation and prediction. This is done this way in order to integrate in future version Cross-validation, Hyper-paramethers
tunning and new information to the original dataset in some processes which need to be developed.
One the training and evaluation have finish start the web application and offers 2 services  

### Prediction:
you need to build a GET request and send it to the server. In these two examples you see the mechanics
Server is listening in localhost IP in port 15433(1 example) or port 15435(example2)

you can use curl to send the request to the server

http://0.0.0.0:15433/1/predictions?main_text=LAVAMAT%20%63479%20%FL%20%A%20%WASCHVOLLAUTOMAT&add_text=WASCHMASCHINEN&manufacturer=AEG

The category predicted is WASHINGMACHINES with a probability of 0.5472955386235383. The algorithm use for preditions is Logistic Regression.
The category predicted is WASHINGMACHINES with a probability of 0.9998689706663867. The algorithm use for preditions is Naive Bayes.

http://127.0.0.1:15435/1/predictions?main_text=UNIVEGA%20%TERRENO%20%10%20%HE%20%MATTBLAUGRAU%20%45%20%CM&add_text=1_7_4&manufacturer=UNIVEGA

The category predicted is BICYCLES with a probability of 0.41772551670338554. The algorithm use for preditions is Logistic Regression.
The category predicted is BICYCLES with a probability of 0.9626641820825794. The algorithm use for preditions is Naive Bayes.


http://127.0.0.1:15435/1/predictions? is the entry point, you can choose the predictor changing the number after the port /1/. 1 means use Logistic Regression , 2 means use Naive Bayes
after predictions? you have the three paramethers:

main_text=UNIVEGA%20%TERRENO%20%10%20%HE%20%MATTBLAUGRAU%20%45%20%CM(Remark change spaces and special simbols for URI quotation mark)
&
add_text=1_7_4
&
manufacturer=UNIVEGA

I took the last 10 records of the dataset and they were not used for training or evaluation so they are completelly new for the predictors. The app answer with the category predicted, the probability of the class and the algoritm use. Notice that there is 4 Classes of products , and I am returning the high probability among the four


H:\Dropbox\python\curl-7.62.0-win64-mingw\bin>curl http://127.0.0.1:5435/1/predictions?main_text=LAVAMAT%20%63479%20%FL%20%A%20%WASCHVOLLAUTOMAT&add_text=WASCHMASCHINEN&manufacturer=AEG
The category predicted is WASHINGMACHINES with a probability of 0.40677060830262707. The algorithm use for preditions is Logistic Regression.

### Accuraccy
It returns the accuraccy of the estimator. From the original Dataset, i removed the last 10 records, then splitted 70% for training and 30% for test sets.

to send a request to the interface you need to build a GET request to the entry point

http://127.0.0.1:5435/1/prediction/accuracy/ 

here same machanic changing the number after the port /1/ you change the predictor. 1 for Logistic Regression, 2 for Naives Bayes

H:\Dropbox\python\curl-7.62.0-win64-mingw\bin>curl http://127.0.0.1:5435/1/prediction/accuracy/

"Accuraccy Logistic Regression estimator: 0.9970021413276231"

H:\Dropbox\python\curl-7.62.0-win64-mingw\bin>curl http://127.0.0.1:5435/2/prediction/accuracy/
"Naives Bayes estimator: 0.8235546038543897"


H:\Dropbox\python\GFK>spark-submit server.py
18/12/02 15:15:33 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
INFO:classification:Starting up the Classification Engine:
INFO:classification:Loading Data Training...
INFO:classification:Preprocessing data...
INFO:classification:First Transformation data...
INFO:classification:Creating grouped objects...
INFO:classification:Second Transformation data. Feature Creation...
INFO:classification:Splitting the data...
Training Dataset Count: 5654
Test Dataset Count: 2335
INFO:classification:Creating Estimators...
INFO:classification:Creating Logistic Regression...
INFO:estimators:Training Logistic Regression...
INFO:estimators:Testing  Logistic Regression...
INFO:estimators:Saving to disk Logistic Regression...
INFO:estimators:Evaluating the model Logistic Regression...


0.9970021413276231


INFO:classification:Creating Naives Bayes...
INFO:estimators:Training Naive Bayes...
INFO:estimators:Testing  Logistic Regression...
INFO:estimators:Saving to disk Naive Bayes...
INFO:estimators:Evaluating the model Naive Baves...


0.8235546038543897


[02/Dec/2018:15:17:54] ENGINE Bus STARTING
INFO:cherrypy.error:[02/Dec/2018:15:17:54] ENGINE Bus STARTING
[02/Dec/2018:15:17:54] ENGINE Started monitor thread 'Autoreloader'.
INFO:cherrypy.error:[02/Dec/2018:15:17:54] ENGINE Started monitor thread 'Autoreloader'.
[02/Dec/2018:15:17:54] ENGINE Serving on http://127.0.0.1:5435
INFO:cherrypy.error:[02/Dec/2018:15:17:54] ENGINE Serving on http://127.0.0.1:5435
[02/Dec/2018:15:17:54] ENGINE Bus STARTED
INFO:cherrypy.error:[02/Dec/2018:15:17:54] ENGINE Bus STARTED


