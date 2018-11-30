from pyspark.sql import SparkSession
import time, sys, cherrypy, os
from paste.translogger import TransLogger
from app import create_app

 
def init_spark_session():
	# load spark session
	spark = SparkSession.builder.appName('GSK_1').getOrCreate()
	# IMPORTANT: pass aditional Python modules to each worker
	spark.sparkContext.setLogLevel("ERROR")
	spark.sparkContext.addPyFile('app.py')
	spark.sparkContext.addPyFile('classification.py')
	spark.sparkContext.addPyFile('estimators.py')
	return spark
 
 
def run_server(app):
 
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)
 
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')
 
    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5432,
        'server.socket_host': '0.0.0.0'
    })
 
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()
 
 
if __name__ == "__main__":
    # Init spark context and load libraries
    sc = init_spark_session()
    dataset_path = os.path.join('datasets', 'ml-latest')
    app = create_app(sc, dataset_path)
 
    # start web server
    run_server(app)