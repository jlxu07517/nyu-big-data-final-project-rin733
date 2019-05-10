#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: recsys training extension 1

Usage:

    $ PYSPARK_PYTHON=$(which python) spark-submit extension_1_modify_count.py hdfs:/user/lj1194/train_sub.parquet

'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main(spark, train_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    train_file : string, path to the parquet file to load
    outputs: train_drop1.parquet --> dropping counts <= 1
             train_drop2.parquet --> dropping counts <= 2
             train_drop3.parquet --> dropping counts <= 3
             train_log2.parquet --> apply log2(counts) transformation
    '''

    train = spark.read.parquet(train_file).repartition(1000).cache()

    train_drop1 = train.withColumn('count', F.when(train['count']==1,0).otherwise(train['count']))
    train_drop2 = train.withColumn('count', F.when(train['count']<=2,0).otherwise(train['count']))
    train_drop3 = train.withColumn('count', F.when(train['count']<=3,0).otherwise(train['count']))

    train.createOrReplaceTempView('train')
    train_log2 = spark.sql('SELECT user_num_id, track_num_id, log2(count) AS count FROM train')

    train_drop1.repartition(1000, ["user_num_id"]).write.parquet('train_drop1.parquet')
    train_drop2.repartition(1000, ["user_num_id"]).write.parquet('train_drop2.parquet')
    train_drop3.repartition(1000, ["user_num_id"]).write.parquet('train_drop3.parquet')
    train_log2.repartition(1000, ["user_num_id"]).write.parquet('train_log2.parquet')

    print('Finished modifying counts data and successfully saved the following files to HDFS:\n')
    print('train_drop1.parquet\ntrain_drop2.parquet\ntrain_drop3.parquet\ntrain_log2.parquet')

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_extension_1').getOrCreate()

    # Get the train set filename from the command line
    train_file = sys.argv[1]

    # Call our main routine
    main(spark, train_file)