# entity_resolution.py
# import re
# import operator
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

conf = SparkConf().setAppName('entity2')
sc = SparkContext.getOrCreate(conf=conf)
sqlCt = SQLContext(sc)


class EntityResolution:
	def __init__(self, dataFile1, dataFile2, stopWordsFile):
		self.f = open(stopWordsFile, "r")
		self.stopWords = set(self.f.read().split("\n"))
		self.stopWordsBC = sc.broadcast(self.stopWords).value
		self.df1 = sqlCt.read.parquet(dataFile1).cache()
		self.df2 = sqlCt.read.parquet(dataFile2).cache()

	def preprocessDF(self, df, cols):
		""" 
		Input: $df represents a DataFrame
			   $cols represents the list of columns (in $df) that will be concatenated and be tokenized

		Output: Return a new DataFrame that adds the "joinKey" column into the input $df

		Comments: The "joinKey" column is a list of tokens, which is generated as follows:
				 (1) concatenate the $cols in $df; 
				 (2) apply the tokenizer to the concatenated string
		Here is how the tokenizer should work:
				 (1) Use "re.split(r'\W+', string)" to split a string into a set of tokens
				 (2) Convert each token to its lower-case
				 (3) Remove stop words
		"""
		def regex_filter(words):
			return[word for word in words if word not in list(sw)]

		sw = self.stopWordsBC
		sw.add('')
		df = df.select(df.id, lower(concat_ws(' ', *cols)).alias('joinKey'))
		df = df.select(df.id, split(df.joinKey, r'\W+').alias('joinKey'))
			
		filter_udf = udf(regex_filter, ArrayType(StringType()))
		df_filtered = df.select(df.id, filter_udf(df.joinKey).alias('joinKey'))
		return(df_filtered)

		

	def filtering(self, df1, df2):
		""" 
		Input: $df1 and $df2 are two input DataFrames, where each of them 
			   has a 'joinKey' column added by the preprocessDF function

		Output: Return a new DataFrame $candDF with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
				where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
				Intuitively, $candDF is the joined result between $df1 and $df2 on the condition that 
				their joinKeys share at least one token. 

		Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons, 
				  you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function. 
				  Please come up with a more efficient algorithm (see hints in Lecture 2). 
		"""
		def flattenThisDF(df):
			df = df.select(df.id, explode(col("joinKey")).alias('joinKey')).cache()
			df = df.filter(length(df.joinKey)>0)
			return(df)

		df1_flat = flattenThisDF(df1).cache()
		df2_flat = flattenThisDF(df2).cache()

		canDF = df1_flat.join(df2_flat, df1_flat.joinKey == df2_flat.joinKey).\
								select(df1_flat.id.alias('id1'), df2_flat.id.alias('id2'))
		canDF = canDF.drop_duplicates().cache()
		canDF = canDF.join(df1, canDF.id1 == df1.id).\
								select(canDF.id1, canDF.id2, df1.joinKey.alias('joinKey1')).cache()
		canDF = canDF.join(df2, canDF.id2 == df2.id).\
								select(canDF.id1, canDF.id2, canDF.joinKey1, df2.joinKey.alias('joinKey2'))
		return (canDF)



	def verification(self, candDF, threshold):
		""" 
			Input: $candDF is the output DataFrame from the 'filtering' function. 
				   $threshold is a float value between (0, 1] 

			Output: Return a new DataFrame $resultDF that represents the ER result. 
					It has five columns: id1, joinKey1, id2, joinKey2, jaccard 

			Comments: There are two differences between $candDF and $resultDF
					  (1) $resultDF adds a new column, called jaccard, which stores the jaccard similarity 
						  between $joinKey1 and $joinKey2
					  (2) $resultDF removes the rows whose jaccard similarity is smaller than $threshold 
		"""
		def jaccard(key1, key2):
			num = len(set(key1).intersection(set(key2)))
			den = len(set(key1).union(set(key2)))
			return float(num/den)
		
		jaccard_udf = udf(jaccard, DoubleType())
		df = candDF.select(candDF.id1, candDF.id2, jaccard_udf(candDF.joinKey1, candDF.joinKey2).alias('jaccard')).cache()
		df = df.filter(df.jaccard >= threshold)
		return(df)



	def evaluate(self, result, groundTruth):
		""" 
		Input: $result is a list of matching pairs identified by the ER algorithm
			   $groundTrueth is a list of matching pairs labeld by humans

		Output: Compute precision, recall, and fmeasure of $result based on $groundTruth, and
				return the evaluation result as a triple: (precision, recall, fmeasure)

		"""
		match = set(result).intersection(groundTruth)
		precision = float(len(match)/len(result))
		recall = float(len(match)/len(groundTruth))
		FMeasure = float((2* precision * recall) / (precision + recall))
		return (precision, recall, FMeasure)


	def jaccardJoin(self, cols1, cols2, threshold):
		newDF1 = self.preprocessDF(self.df1, cols1)
		newDF2 = self.preprocessDF(self.df2, cols2)
		print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()) )

		candDF = self.filtering(newDF1, newDF2).cache()
		print ("After Filtering: %d pairs left" %(candDF.count()))

		resultDF = self.verification(candDF, threshold).cache()
		print ("After Verification: %d similar pairs" %(resultDF.count()))

		return resultDF


	def __del__(self):
		self.f.close()


if __name__ == "__main__":
	# dir = '../../amazon-google-sample/'
	# er = EntityResolution( dir+"Amazon_sample", dir+"Google_sample", dir+"stopwords.txt")
	
	# big sample
	dir = '../../amazon-google-big/'
	er = EntityResolution( dir+"Amazon", dir+"Google", dir+"stopwords.txt")

	amazonCols = ["title", "manufacturer"]
	googleCols = ["name", "manufacturer"]
	resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

	result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
	groundTruth = sqlCt.read.parquet(dir + "Amazon_Google_perfectMapping") \
	                      .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
	print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
