#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' recsys fit indexer

Usage:

    $ spark-submit fit_indexer_int.py ./train_downsample.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet

'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType

def main(spark, train_file, val_file, test_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    train_file : string, path to the train parquet file to load
    val_file : string, path to the val parquet file to load
    test_file : string, path to the test parquet file to load

    '''

    train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    test = spark.read.parquet(test_file)

    indexer1 = StringIndexer(inputCol="user_id", outputCol="user_num_id").setHandleInvalid("skip")
    indexer2 = StringIndexer(inputCol="track_id", outputCol="track_num_id").setHandleInvalid("skip")
    pipeline = Pipeline(stages=[indexer1, indexer2])

    idx = pipeline.fit(train)
    train = idx.transform(train).select(['user_num_id','track_num_id','count'])
    val = idx.transform(val).select(['user_num_id','track_num_id','count'])
    test = idx.transform(test).select(['user_num_id','track_num_id','count'])

    # count: val = 133,628,  test = 1,344,207, train = 3,868,582 (downsampled)

    val = val.withColumn("user_num_id", val["user_num_id"].cast(IntegerType()))
    val = val.withColumn("track_num_id", val["track_num_id"].cast(IntegerType()))
    val.write.parquet("val_sub.parquet")

    test = test.withColumn("user_num_id", test["user_num_id"].cast(IntegerType()))
    test = test.withColumn("track_num_id", test["track_num_id"].cast(IntegerType()))
    test.write.parquet("test_sub.parquet")

    train = train.withColumn("user_num_id", train["user_num_id"].cast(IntegerType()))
    train = train.withColumn("track_num_id", train["track_num_id"].cast(IntegerType()))
    train.repartition(1000, "user_num_id").write.parquet('train_sub.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_fit_indexer').getOrCreate()

    # Get the train set filename from the command line
    train_file = sys.argv[1]

    # Get the val and test set filename from the command line
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, test_file)
