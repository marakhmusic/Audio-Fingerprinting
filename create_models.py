import numpy
import pandas as pd
import config
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
# Importing libraries for building the neural network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import time
import pickle
from models import create_baseline
import argparse
from keras.utils import np_utils
from termcolor import colored


# Import the training data set
def import_training_data(file_name=config.TRAINING_FILENAME):
    data_set = pd.read_csv(file_name)
    return data_set


def save_model(model):
    parser = argparse.ArgumentParser(description='Create a ML model to predict the phase change time positions')
    parser.add_argument('-f', '--file_name', help="file name", type=str)
    args =  parser.parse_args()
    file_name = args.file_name
    try:
        pickle.dump(model, open(file_name, 'wb'))
        print(f'pipeline model has been saved as {file_name}')
    except Exception as e:
        print(f'Error while dumping model in a file: {e}')



if __name__ == '__main__':
    seed = 7
    numpy.random.seed(seed)            # for Reproducibility

    begin_time = time.time()

    # Import the data_set from the training file
    data_set = import_training_data()

    # Print top 5 rows
    print(data_set.head())

    # Define the variables and the output columns
    X = data_set.loc[:, data_set.columns != 'label']
    Y = data_set.loc[:, data_set.columns == 'label']

    # Encode Y values
    # encoder = LabelEncoder()
    # encoder.fit(Y)
    # encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    #dummy_y = np_utils.to_categorical(encoded_Y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(Y)
    #print('encoded_Y', encoded_Y)
    #print('dummy_y', dummy_y)
    print('y', y)


    # Evaluate model using standardized dataset.
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=1000, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)

    print('using StratifiedKFold for evaluating accuracy')
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    # Fit the model on the given data_set
    pipeline.fit(X,y)
    print('pipeline has been fitted')

    save_model(pipeline)            # Save the model in a file


    model_ready_time = time.time()
    time_taken_in_fitting = model_ready_time - begin_time
    print(f'time taken in readying model {time_taken_in_fitting}')

    # Predict for all the items present in the dataset
    # counter = 0
    # for index, row in X.iterrows():
    #     prediction = pipeline.predict([row])[0]
    #     if prediction == 1:
    #         print(colored(counter,'green'))
    #     elif prediction  == 0:
    #         print(colored(counter,'yellow'))
    #     elif prediction == 2:
    #         print(colored(counter,'grey'))
    #     counter  +=1
    # print()
    #
    # end_time = time.time()
    # print(f'total time taken in predictions: {end_time - model_ready_time} ')
    # print(f'total time taken by script: {end_time - begin_time} ')
