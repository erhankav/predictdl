"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask,request,jsonify
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
import numpy as np
import ast


app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
def hello():
   
    filename = 'iris.csv'
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(filename, names=names)
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    # save the model to disk
    filename = 'finalized_model.pkl'
    dump(knn, filename)
    return "Hello World!"

@app.route('/predict', methods=['GET'])
def get_product():

    filename = 'finalized_model.pkl'
    loaded_model = load(filename)
    dizi=request.args.get('dnm');
    dnm=ast.literal_eval(dizi);
    npdizi=np.array(dnm)
    #result = loaded_model.predict([[dizi[0],dizi[0],dizi[0],dizi[0]]])
    return str(npdizi)
    #return jsonify({'prediction': list(result)})

    
#@app.route("/predict", methods=['GET'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
    #app.run(debug=True)




