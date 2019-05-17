#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' recsys downsample

Usage:

    $ PYSPARK_PYTHON=$(which python) spark-submit downsample_train_set.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet hdfs:/user/bm106/pub/project/cf_test.parquet

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

def main(spark, train_file, val_file, test_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    train_file : string, path to the original train parquet file to load
    val_file : string, path to the originalval parquet file to load
    test_file : string, path to the originaltest parquet file to load

    '''
    train = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    train.createOrReplaceTempView('train')

    val = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    val.createOrReplaceTempView('val')

    test = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')
    test.createOrReplaceTempView('test')

    data1 = spark.sql('select * from train where user_id in (SELECT user_id FROM test) or user_id in (SELECT user_id FROM val)')
    data1.createOrReplaceTempView('data1')
    data2 = train.sample(False, 0.05, 1)
    dfUnion = data1.union(data2)
    df_train = dfUnion.dropDuplicates()
    df_train.repartition(1000, "user_id").write.parquet("train_downsample.parquet")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_downsample').getOrCreate()

    # Get the train set filename from the command line
    train_file = sys.argv[1]

    # Get the val and test set filename from the command line
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, val_file, test_file)
