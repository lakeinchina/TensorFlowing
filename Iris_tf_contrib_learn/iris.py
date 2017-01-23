# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

IRIS_TRAING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

traing_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="tmp/iris_model")
classifier.fit(x=traing_set.data,
               y=traing_set.target,
               steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                    y=test_set.target)["accuracy"]
print "Acc=%f"%(accuracy_score)