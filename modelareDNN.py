from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
matches = matches.reindex(
       np.random.permutation(matches.index))
       
def preprocess_features(matches):
    
    selected_features = matches[["average_rank", "rank_difference", "point_difference", "is_stake"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(matches):
    output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
    output_targets["is_won"] = matches['is_won']
    return output_targets
    
# Alegem primele din date 60% ex: 10900 (din 18167) exemple pentru antrenare.
training_examples = preprocess_features(matches.head(10900))
training_targets = preprocess_targets(matches.head(10900))

# Restu 40%.
validation_examples = preprocess_features(matches.tail(7267))
validation_targets = preprocess_targets(matches.tail(7267))

Complete_Data_training = preprocess_features(matches)
Complete_Data_Validation = preprocess_targets(matches)

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
              
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
     # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
    
def train_nn_classification_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 10
    steps_per_period = steps / periods
  # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 3.0)
    dnn_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["is_won"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["is_won"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
    # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
    # Train the model, starting from the prior state.
        dnn_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
        # Take a break and compute predictions.    
        training_probabilities = dnn_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
        validation_probabilities = dnn_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")
      # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return dnn_classifier
    
linear_classifier = train_nn_classification_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.07),
    steps=3000,
    batch_size=2000,
    hidden_units=[5, 5,6,5],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)

# Validarea Modelului

validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class.
validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)


evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])


def train_complete_model(my_optimizer,
    steps,
    batch_size,
    hidden_units,
    Complete_Data_training,
    Complete_Data_Validation) :
    
    periods = 10
    steps_per_period = steps / periods
  # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 3.0)
    dnn_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(Complete_Data_training),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
    # Create input functions.
    Complete_training_input_fn = lambda: my_input_fn(Complete_Data_training, 
                                          Complete_Data_Validation["is_won"], 
                                          batch_size=batch_size)
    predict_Complete_training_input_fn = lambda: my_input_fn(Complete_Data_training, 
                                                  Complete_Data_Validation["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    # validation_log_losses = []
    for period in range (0, periods):
    # Train the model, starting from the prior state.
        dnn_classifier.train(
        input_fn=Complete_training_input_fn,
        steps=steps_per_period
    )
        # Take a break and compute predictions.    
        Complete_training_probabilities = dnn_classifier.predict(input_fn=predict_Complete_training_input_fn)
        Complete_training_probabilities = np.array([item['probabilities'] for item in Complete_training_probabilities])
    
        
        training_log_loss = metrics.log_loss(Complete_Data_Validation, Complete_training_probabilities)
        #validation_log_loss = metrics.log_loss(Complete_Data_Validation, validation_probabilities)
    # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        #validation_log_losses.append(validation_log_loss)
    print("Model training finished.")
      # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    #plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return dnn_classifier
    
linear_classifier = train_complete_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.07),
    steps=3000,
    batch_size=2000,
    hidden_units=[5, 5,6,5],
    Complete_Data_training=Complete_Data_training,
    Complete_Data_Validation=Complete_Data_Validation)
    
# Grupele
margin = 0.05

# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                    rankings['country_full'].isin(world_cup.index.unique())]
                                    

world_cup_rankings = world_cup_rankings.set_index(['country_full'])

world_cup_rankings.head()

