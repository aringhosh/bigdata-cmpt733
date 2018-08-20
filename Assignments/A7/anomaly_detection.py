# anomaly_detection.py
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator

from pyspark.sql import SQLContext

spark = SparkSession.builder.appName('app').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)



class AnomalyDetection():
    
    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]), \
                (1, ["http", "udf", 0.5]), \
                (2, ["http", "tcp", 0.5]), \
                (3, ["ftp", "icmp", 0.1]), \
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = sqlContext.createDataFrame(data, schema)
        
    def readData(self, filename):
        self.rawDF = spark.read.parquet(filename).cache()
    
    def cat2Num(self, df, indices):
        """
            Write your code!
        """
        self.key_dict = dict()
        new_df = df.select('rawFeatures')
        for idx in indices: 
            new_col_name = str(idx)
            new_df = new_df.withColumn(new_col_name, df.rawFeatures[idx])
            self.key_dict[idx] = [x[new_col_name] for x in new_df.select(new_col_name).distinct().collect()] # since the list is small
        
        self.bc_unique_dict = sc.broadcast(self.key_dict).value
        bc_unique_dict_local = self.bc_unique_dict
        
        def convert(this_column):
            new_column = this_column
            
            for i in indices:
                unique_values = bc_unique_dict_local[i]
                val = this_column[i]
                encoded = list()
                for v in unique_values:
                    encoded.append(1) if v == val else encoded.append(0)
                new_column[i] = encoded
            
            
            onehot_list = [i for j, i in enumerate(this_column) if j in indices]
            onehot_list = [y for x in onehot_list for y in x]
            non_cat_list = [i for j, i in enumerate(this_column) if j not in indices]
            return onehot_list + non_cat_list
            
        
        convert_udf = udf(convert, ArrayType(StringType()))
        df = df.select(df['id'], df['rawFeatures'], convert_udf(df['rawFeatures']).alias('features'))
        return df
        
        

    def addScore(self, df):
        """
            Write your code!
        """
        count_table = df.groupBy('prediction').count()
        count_table = count_table.withColumnRenamed('count', 'cluster_count')
        n_max = int(count_table.agg({"cluster_count": "max"}).collect()[0]["max(cluster_count)"])
        n_min = int(count_table.agg({"cluster_count": "min"}).collect()[0]["min(cluster_count)"])
        
        if(n_min == n_max) : # division by zero; set score 0
            df.withColumn('score', 0)
        else :
            df = df.join(count_table, df.prediction == count_table.prediction).drop(df.prediction)
            df = df.withColumn('score', (n_max - df.cluster_count)/ (n_max - n_min))
        
        df = df.drop(df.cluster_count)
        return df
         
    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()
        
        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)
        
        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()
        
        #Adding the score column to df2; The higher the score, the more likely it is an anomaly 
        df3 = self.addScore(df2).cache()
        df3.show()    
      
        return df3.where(df3.score > t)
    
 
if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData('data/logs-features-sample')
    anomalies = ad.detect(8, 0.97)
    print (anomalies.count())
    anomalies.show()
    
    
#   # test - toy data
#     ad.readToyData()
#     ad.cat2Num(ad.rawDF, [0,1])
#     anomalies = ad.detect(2, 0.9)
#     print (anomalies.count())
#     anomalies.show()