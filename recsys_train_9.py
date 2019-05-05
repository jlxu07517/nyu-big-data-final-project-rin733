#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: recsys training

Usage:

    $ PYSPARK_PYTHON=$(which python) spark-submit recsys_train_9.py hdfs:/user/lj1194/train_sub.parquet hdfs:/user/lj1194/val_sub.parquet ./recsys_model_9 > output9_1.txt

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

    train = spark.read.parquet(train_file).repartition(5000, ["user_num_id", "count"]).cache()
    # train.createOrReplaceTempView('train')
    val = spark.read.parquet(val_file)
    val.createOrReplaceTempView('val')
    val_users = val.select('user_num_id').distinct().cache()
    truth = spark.sql('SELECT user_num_id AS user_id, collect_list(track_num_id) AS label FROM val GROUP BY user_num_id').repartition(1000, "user_id").cache()

    rank_list = [90, 110, 120, 130, 140, 150]
    regParam_list = [0.1, 0.05, 0.01]
    alpha_list = [0.5, 1, 2]
    metric_list = []
    best_model = None
    best_metric = 0
    best_rank = None
    best_regParam = None
    best_alpha = None

    for rank, regParam, alpha in itertools.product(rank_list, regParam_list, alpha_list):
    # for rank in rank_list:
        als = ALS(rank=rank, regParam=regParam, alpha=alpha, maxIter=10, userCol="user_num_id", itemCol="track_num_id",
            ratingCol="count", implicitPrefs=True, coldStartStrategy="drop")
        model = als.fit(train)
        # get recommendations
        userSubsetRecs = model.recommendForUserSubset(val_users, 500)
        recs = userSubsetRecs.select("recommendations.track_num_id","user_num_id")
        # get input for ranking metrics
        pred = truth.join(recs, truth.user_id == recs.user_num_id, how='left').select('track_num_id', 'label')
        predictionAndLabels = pred.rdd.map(lambda lp: (lp.track_num_id, lp.label)).repartition(100)

        print('--------Start Computing rank = {}, regParam = {}, alpha = {}... ...--------'.format(rank, regParam, alpha))
        metrics = RankingMetrics(predictionAndLabels)
        meanAP = metrics.meanAveragePrecision
        metric_list.append(meanAP)
        print('Mean Average Precision of rank/regParam/alpha {}/{}/{} = {}'.format(rank, regParam, alpha, meanAP))
        print('--------Finish Computing--------\n')

        if (meanAP > best_metric):
            best_model = model
            best_metric = meanAP
            best_rank = rank
            best_regParam = regParam
            best_alpha = alpha

    print("The best model has meanAP = {} : rank = {}, regParam = {}, alpha = {}".format(str(best_metric), best_rank, best_regParam, best_alpha))
    best_model.write().overwrite().save(model_file)


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