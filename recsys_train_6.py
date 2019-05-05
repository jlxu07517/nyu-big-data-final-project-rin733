#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: recsys training

Usage:

    $ nohup spark-submit recsys_train_6.py ./train_downsample.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet ./recsys_6

'''


# We need sys to get the command line arguments
import sys
import itertools

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, train_file, val_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    train = spark.read.parquet(train_file)
    # train.createOrReplaceTempView('train')
    val = spark.read.parquet(val_file)

    indexer1 = StringIndexer(inputCol="user_id", outputCol="user_num_id").setHandleInvalid("skip")
    indexer2 = StringIndexer(inputCol="track_id", outputCol="track_num_id").setHandleInvalid("skip")
    pipeline = Pipeline(stages=[indexer1, indexer2])

    idx = pipeline.fit(train)
    train = idx.transform(train).select(['user_num_id','track_num_id','count'])
    val = idx.transform(val).select(['user_num_id','track_num_id','count'])

    val.createOrReplaceTempView('val')
    val_users = val.select('user_num_id').distinct()

    als = ALS(maxIter=5, rank=10, regParam=0.1, userCol="user_num_id", itemCol="track_num_id", ratingCol="count",
        implicitPrefs=True, alpha=1, coldStartStrategy="drop")
    model = als.fit(train)

    userSubsetRecs = model.recommendForUserSubset(val_users, 500)
    recs = userSubsetRecs.select("recommendations.track_num_id","user_num_id")

    truth = spark.sql('SELECT user_num_id AS user_id, collect_list(track_num_id) AS label FROM val GROUP BY user_num_id')

    pred = truth.join(recs, truth.user_id == recs.user_num_id, how='left').select('track_num_id', 'label')
    predictionAndLabels = pred.rdd.map(lambda lp: (lp.track_num_id, lp.label))

    print('++++++')
    metrics = RankingMetrics(predictionAndLabels)
    print(metrics.meanAveragePrecision)
    print('++++++')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_train').getOrCreate()

    # Get the train set filename from the command line
    train_file = sys.argv[1]

    # Get the val set filename from the command line
    val_file = sys.argv[2]

    # And the location to store the trained model
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, model_file)