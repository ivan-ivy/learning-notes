## Big Data Fundamentals with PySpark
Two mods for spark: local and cluster
SparkContext is the entry point to creating RDDs.

Three ways to load data into RDDs:
* sc.parallelize()
* From external datasets, e.g. sc.textFile()

Operations on RDDs:
* Transformations (lazy evaluation): map(), filter(), flatMap(), union()
* Actions: collet(), take(), first(), count()

Pair RDD: reduceByKey(), groupByKey(), sortByKey(), join(), countByKey(), collectAsMap()

Reduce() action:

reduce action is used for aggregating the elements of a regular RDD.
The function should be commutative and associative

saveAsTextFile() saves RDD  a text file inside a directory with each partition as a separate file. use `coalesce` to save RDD as a single text file.

deal with unstructured data
```python
# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)

# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split(' '))
print("Total number of words in splitRDD:", splitRDD.count())

# Convert the words in lower case and remove stop words from stop_words
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Count of the number of occurences of each word
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)

# Display the first 10 words and their frequencies
for word in resultRDD.take(10):
	print(word)

# Show the top 10 most frequent words and their frequencies
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[0])) # swap key and value
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)
for word in resultRDD_swap_sort.take(10):
	print("{} has {} counts". format(word[1], word[0]))
```

### DataFrames
SparkSession provides a single point of entry to interact with Spark DatFrames. it is `spark` in shell

Two way to create DataFrames:
* use `.createDataFrame()` on existing RDDs
* use `.read()` on various data sources

```python
# use createDataFrame
sample_list = [('Mona',20), ('Jennifer',34), ('John',20), ('Jim',26)]
rdd = sc.parallelize(sample_list)
names_df = spark.createDataFrame(rdd, schema=['Name', 'Age'])

# use read
people_df = spark.read.csv(file_path, header=True, inferSchema=True)
```

DataFrame Transformations:(create a new DataFrame)
* `select()`: subsets the columns in the DataFrame
* `filter()`: filters out the rows based on a condition
* `groupby()`: used to group a variable
* `orderby()`: sorts the DF based one or more columns
* `dropDuplicates()`: remove duplicate rows of a DataFrame
* `withColumnRenamed()`: renames a column in the DataFrame

DataFrame Actions:
* `printSchema()`: prints the types of columns in the DataFrame
* head(): 
* show
* count
* columns: print the columns of the df
* describe(): compute summary statistics of numerical columns 

With SQL:
* df.createOrReplaceTempView()
* spark.sql(): run SQL query
  
Data Visualization
* pyspark_dist_explore lib: `hist()`, `distplot()`, `pandas_histogram()`
* toPandas()
* HandySpark lib


### PySpark MLlib
3 C's:
pyspark.mllib.recommendation
pyspark.mllib.classification
pyspark.mllib.clustering

Example
```python
# Collaborative filtering
# Load the data 
data = sc.textFile(file_path)
ratings = data.map(lambda l: l.split(','))
ratings_final = ratings.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))

# Split the data into training and test
training_data, test_data = ratings_final.randomSplit([0.8, 0.2])
# Create the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)

testdata_no_rating = test_data.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata_no_rating)

# Evaluation
rates = ratings_final.map(lambda r: ((r[0], r[1]), r[2]))
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = rates.join(preds)

# Calculate and print MSE
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))
```





Logistic Regression
```python
# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))


# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))
samples = spam_samples.union(non_spam_samples)

# Split the data into training and testing
train_samples,test_samples = samples.randomSplit([0.8, 0.2])

model = LogisticRegressionWithLBFGS.train(train_samples)
predictions = model.predict(test_samples.map(lambda x: x.features))
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))
```

Clustering
```python
# load data
clusterRDD = sc.textFile(file_path)
rdd_split = clusterRDD.map(lambda x: x.split('\t'))
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])

# trian model
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k 
model = KMeans.train(rdd_split_int, k=15, seed=1)

cluster_centers = model.clusterCenters

# visualization
rdd_split_int_df = spark.createDataFrame(rdd_split_int, schema=["col1", "col2"])
rdd_split_int_df_pandas = rdd_split_int_df.toPandas()
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])

# Create an overlaid scatter plot
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()
```