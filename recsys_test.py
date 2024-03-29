#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: recsys testing
Usage:
    $ spark-submit recsys_test.py hdfs:/path/to/load/best_model.parquet hdfs:/user/lj1194/test_sub.parquet
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, model_file, data_file):
    '''
    Parameters
    ----------
    spark : SparkSession object
    model_file : string, path to the best model file
    data_file : string, path to the test parquet file to load
    '''

    # Loads test data
    data = spark.read.parquet(data_file).repartition(5000, "user_num_id")
    data.createOrReplaceTempView('data')

    # Loads trained ALS model
    model = ALSModel.load(model_file)

    users = data.select('user_num_id').distinct()
    truth = spark.sql('SELECT user_num_id AS user_id, collect_list(track_num_id) AS label FROM data GROUP BY user_num_id')

    # get recommendations
    userSubsetRecs = model.recommendForUserSubset(users, 500)
    recs = userSubsetRecs.select("recommendations.track_num_id","user_num_id")
    # get input for ranking metrics
    pred = truth.join(recs, truth.user_id == recs.user_num_id, how='left').select('track_num_id', 'label')
    predictionAndLabels = pred.rdd.map(lambda lp: (lp.track_num_id, lp.label))

    print('--------Start Computing ... ...--------')
    metrics = RankingMetrics(predictionAndLabels)
    meanAP = metrics.meanAveragePrecision
    print('Mean Average Precision on test set = {}'.format(meanAP))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_test').getOrCreate()

    # And the location to our trained model
    model_file = sys.argv[1]

    # Get the test filename from the command line
    data_file = sys.argv[2]

    # Call our main routine
    main(spark, model_file, data_file)