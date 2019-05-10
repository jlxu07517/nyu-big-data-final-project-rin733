#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 3: Fast search benchmarking
Task: For all users, get the top 500 relevant items
1. Get the user factors for queries.
2. Prepare the user factor and item factor for similarity search
3. Run the exact search , record time and MAP 
4. Run the approximate search ,record time and MAP

# Should we compare in MAP(of the orginal task or should we compare its precision with respect to the rank list returned by exact search?)
# Do I include the process of preparing data for nmslib in build time?
# Do I only care about training time?
# There is a tradeoff between build time , retrieval time and recall/precision
Usage:

    $ spark-submit recsys_train_1.py hdfs:/user/bm106/pub/project/cf_validation.parquet ./recsys_1

'''


# We need sys to get the command line arguments
import sys
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import nmslib
import numpy as np

class nmslib_search:

    def __init__(self,index_params=None,
                 query_params=None, print_progress=False):
       
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def buildindex(self,itemfactor,itemindex):
        '''
           Run this to build index on itemfactors before you run quereis!'''
        index_params = self.index_params
        

        if index_params is None:
            index_params = {'M':16,'post':0,'efConstruction': 400}
        

        index = nmslib.init(space='cosinesimil', method='hnsw')
        index.addDataPointBatch(itemfactor,ids = itemindex)
        index.createIndex(index_params,self.print_progress)
        

        return index

    def query(self,index,userfactor,topk):
        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        allnb = index.knnQueryBatch(userfactor,k=topk,num_threads = 4)
        #allitems = [each[0] for each in allnb]
        #allnb is a list of length len(userfactor)
        #allnb[i] is (itemindexes,distances in increasing order)
        return allnb #a list of lists



def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : Utility dataframe for testing/queries, include the user_num_id column
    model_file : saved ALSModel
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    # Loads data and model
    data = spark.read.parquet(data_file)
    model = ALSModel.load(model_file)

    data.createOrReplaceTempView('val')
    val_users = data.select("user_num_id").distinct()
    #labels
    #truth  = spark.sql('SELECT user_num_id AS user_id, collect_list(track_num_id) AS label FROM val GROUP BY user_num_id').repartition(1000, "user_id")
    start = time.time()
    userSubsetRecs = model.recommendForUserSubset(val_users, 500)
    exact_qtime =  time.time() - start
    #exact recs
    #recs = userSubsetRecs.selectExpr("recommendations.track_num_id as exact_rec","user_num_id")
    #allres = truth.join(recs, truth.user_id == recs.user_num_id, how='left') #excact recommend track and label, user_id and user_num_id
 
    #bu ji
    #exactpred = allres.select('track_num_id', 'label')

    #prepare data from nmslib query
    userfactor = val_users.join(model.userFactors ,val_users.user_num_id == model.userFactors.id,how = 'left').select('id','features') #keep id to check whether the same as in allres
    #Get the user where we have representation in the model, hopefully it covers all users in the validation set and test set.
    queries = np.array([row.features for row in userfactor.select('features').collect()])
    queries = np.append(queries,np.zeros((queries.shape[0],1)),axis = 1)
    
    #Prepare itemfactors
    itemfactors = np.array([row.features for row in model.itemFactors.select('features').collect()])
    itemidx = np.array([row.id for row in model.itemFactors.select('id').collect()])
    
    norms = np.linalg.norm(itemfactors,axis = 1)
    maxnorm = norms.max()
    extra_item_dim = np.sqrt(maxnorm ** 2 - norms ** 2)
    itemfactors = np.append(itemfactors, extra_item_dim.reshape(norms.shape[0], 1), axis=1)
    #allset

    nms = nmslib_search()
    nmsindex = nms.buildindex(itemfactors,itemidx)
    start = time.time()
    res = nms.query(nmsindex,queries,500)
    approx_qtime = time.time() - start
    

    print('Time to search top 500 exactly is {}; Time to search using nmslib is {}'.format(exact_qtime,approx_qtime))

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recsys_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)