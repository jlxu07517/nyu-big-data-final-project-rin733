#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: recsys training
Usage:
    PYSPARK_PYTHON=$(which python) spark-submit lsh_nmslib.py ./distinct_test_users.parquet ./recsys_model_search ./test_truth.parquet >output_lsh.txt
'''


# We need sys to get the command line arguments
import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import nmslib
import numpy as np
import time
from pyspark.sql import Row
import itertools

def bruteforce(user,item,itemidx,topk):
    '''
    user: transformed user array of shape(num_users,rank)
    item:transformed item array of shaope (num_items,rank)
    itemidx: the original item indeces, 1d numpy array of length num_item
    topk: searching for top k nearest neighbors
    '''
    residx = []
    resdist = []
    k = 0
    while k + 500 < len(user):    
        pairwise_dot = np.matmul(user[k:k+500,:],item.T) #shape = (num_users,num_times)
        desc_index = np.argsort(pairwise_dot,axis = 1)[:, ::-1][:,:topk] #shape = (num_users,num_times)
        residx.extend(itemidx[desc_index].tolist())
        resdist.extend([pairwise_dot[i][desc_index[i]].tolist() for i in range(len(pairwise_dot))])
        k += 500
    if k < len(user):
        pairwise_dot = np.matmul(user[k:,:],item.T) #shape = (num_users,num_times)
        desc_index = np.argsort(pairwise_dot,axis = 1)[:, ::-1][:,:topk] #shape = (num_users,num_times)
        residx.extend(itemidx[desc_index].tolist())
        resdist.extend([pairwise_dot[i][desc_index[i]].tolist() for i in range(len(pairwise_dot))])
        
    return residx,resdist

def nmslib_search(index, trs_user,trs_item,itemidx,topk = 5,efs = 1000):
    '''trs_user: transformed user factor matrix
        trs_item:transformed item factor matrix,
        itemidx: index of each item'''
    queryParams = {'efSearch': efs}
    index.setQueryTimeParams(queryParams)
    res = index.knnQueryBatch(trs_user,k=topk,num_threads = 4)
    residx = [each[0].tolist() for each in res] #Recommended indices
    resdist = [each[1].tolist() for each in res] #Distance: Smaller means larger inner product
    return residx,resdist


def main(spark, data_file, model_file,truth_file):
    '''Main routine for supervised training
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load: test set user id.
    model_file : string, path to store the serialized model file
    truth_file: ground truth interaction list for each user in the test set.
    '''

    topk =  500
    #nmslib indexing parameters
    Mlist = [50]
    efclist = [3000]
    efslist = [800]

    #Prepare data
    test_users = spark.read.parquet(data_file) #distinct_test_users.parquet
    truth = spark.read.parquet(truth_file)
    model = ALSModel.load(model_file) #recsys_model_search
    usefactor = model.userFactors
    queryuser = test_users.join(usefactor,test_users.user_num_id == usefactor.id,how = 'left')
    userid = [row.id for row in queryuser.select('id').collect()]
    user = np.array([row.features for row in queryuser.select('features').collect()])
    itmfactor = model.itemFactors
    item = np.array([row.features for row in itmfactor.select('features').collect()])
    
    itemidx = np.array([row.id for row in itmfactor.select('id').collect()])
    trs_user = np.append(user,np.zeros((user.shape[0],1)),axis = 1)
    norms = np.linalg.norm(item,axis = 1)
    maxnorm = norms.max()
    extra_item_dim = np.sqrt(maxnorm ** 2 - norms ** 2)
    trs_item = np.append(item, extra_item_dim.reshape(norms.shape[0], 1), axis=1)
    print('Finish Preparing the data')
    print('Start brute force search')
    #only try brute force once.
    #time2 = time.time()
    #brutea,bruteb = bruteforce(user,item,itemidx,topk)
    #brute_time = time.time() - time2
    #print('Time to brute force search top{} items is {}, {} seconds per query'.format(topk,brute_time,brute_time/len(user)))
    
    #Get MAP
    R = Row('id', 'recs')
    #rec_brute = spark.createDataFrame([R(x, y) for i, (x,y) in enumerate(zip(userid,brutea))])
    #pred_brute = truth.join(rec_brute, truth.user_id == rec_brute.id, how='left').select('recs', 'label')
    #predictionAndLabels_b = pred_brute.rdd.map(lambda lp: (lp.recs, lp.label)).repartition(100)
    #metrics_b = RankingMetrics(predictionAndLabels_b)
    #meanAP_b = metrics_b.meanAveragePrecision



   #Multiple accelerated search with different parameter settings
    for M,efc,efs in itertools.product(Mlist,efclist,efslist):
        indexParams = {'M': M, 'indexThreadQty': 4, 'efConstruction': efc, 'post' : 0}
    
        #Get time
        print("__________________Start a new indexer______________________")
        index = nmslib.init(method = 'hnsw',space = 'cosinesimil')
        index.addDataPointBatch(trs_item,ids = itemidx)
        time1 = time.time()
        index.createIndex(indexParams)
        nmslib_buildtime = time.time() - time1
        print('indexParams for nmslib is {}'.format(indexParams),'queryParams for nmslib is efs =  {}'.format(efs))
        print('Time to build index for nmslib is {}'.format(nmslib_buildtime))
       
        time3 = time.time()
        nms_a,nms_b = nmslib_search(index, trs_user,trs_item,itemidx,topk,efs)
        nms_time = time.time() - time3
        print('Time to nmslib search top{} items is {}, {} seconds per query'.format(topk,nms_time,nms_time/len(user)))

        #Get MAP
        rec_nms = spark.createDataFrame([R(x,y) for i ,(x,y) in enumerate(zip(userid,nms_a))])
        pred_nms = truth.join(rec_nms,truth.user_id == rec_nms.id,how = 'left').select('recs','label')
        predictionAndLabels_n = pred_nms.rdd.map(lambda lp: (lp.recs, lp.label)).repartition(100) 
        metrics_n = RankingMetrics(predictionAndLabels_n)
        meanAP_n = metrics_n.meanAveragePrecision

        print(' MAP for nmslib is {},MAP for bruteforce is 0.04126000613271917'.format(meanAP_n))
        print('____________Finish an indexer__________________')
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('lsh').getOrCreate()

    # Get the train set filename from the command line
    data_file = sys.argv[1]

    # Get the val set filename from the command line
    model_file = sys.argv[2]

    # And the location to store the trained model
    truth_file = sys.argv[3]

    # Call our main routine
    main(spark, data_file, model_file, truth_file)

