## Intro to Spark
Spark is a platform for cluster computing. Spark lets you spread data and computations over clusters with multiple nodes.

Two mods:
* Clusters
* local(will show local in sc.master)
### Using Spark in Python

First step in using Spark is connecting to a cluster.
* Master: manages splitting up the data and the computations
* Worker: rest of the computers in the cluster

PySpark:
`SparkContext`: create the connection. call it sc
`SparkConf()`: hold all arguments and attributes

### DataFrames
Spark's core data structure is the Resilient Distributed Dataset (RDD). We can use Spark DataFrame abstraction built on top of RDDs. DataFrame is like a SQL table, they are also more optimized for complicated operations than RDDs.


Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the SparkSession.builder.getOrCreate() method. This returns an existing SparkSession if there's already one in the environment, or creates a new one if necessary.

```python
from pyspark.sql import SparkSession

# crate session
spark = SparkSession.builder.getOrCreate()
print(spark.catalog.listTables())

# show table
query = "FROM flights SELECT * LIMIT 10"
flights10 = spark.sql(query)
flights10.show()

# to pandas data frame
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
flight_counts = spark.sql(query)
pd_counts = flight_counts.toPandas()

# create dataframe and add it to spark
pd_temp = pd.DataFrame(np.random.random(10))
spark_temp = spark.createDataFrame(pd_temp)
spark_temp.createOrReplaceTempView("temp")
print(spark.catalog.listTables())

# create dataframe from txt file
file_path = "/usr/local/share/datasets/airports.csv"
airports = spark.read.csv(file_path, header=True)
airports.show()
```

`SparkSession.catalog`: lists all the data inside the cluster
`SparkSession.catalog.listTables()`: return the names of all the tables as a list
`SparkSession.sql()`: run sql query on tables
`SparkSession.sql().toPandas()`: convert table to a pandas dataframe

`SparkSession.createDataFrame()`: create dataframe from pandas dataframe, and it is stored locally. use `.createOrReplaceTempView()` to add this df to a temporary table in this session


### Manipulating Data
```python
# creating columns
flights = spark.table("flights")
flights = flights.withColumn("duration_hrs", flights.air_time/60)

# filtering data
long_flights1 = flights.filter("distance > 1000")
long_flights2 = flights.filter(flights.distance > 1000)

# select
selected1 = flights.select('tailnum', 'origin', 'dest')
temp = flights.select(flights.origin, flights.dest, flights.carrier)
filterA = flights.origin == "SEA"
filterB = flights.dest == "PDX"
selected2 = temp.filter(filterA).filter(filterB)

# select 2
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

# Group by
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()
flights.filter(flights.carrier=='DL').filter(flights.origin=='SEA').groupBy().avg('air_time').show()
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

# rename faa to dest
airports = airports.withColumnRenamed("faa", "dest")

# join 
flights_with_airports = flights.join(airports, on='dest', how='leftouter')
```
Spark DataFrame is immutable. To update, just reassign the returned DataFrame.

`filter`: To filter the data, it is similar to where clause in SQL. Just pass a string to the function or a column of boolean values.

`select`: can use column name as string or df.colName. With df.colName, you can perform any column operation and alias() to rename. This is equivalent to `.selectExpr()` with SQL expressions.

`groupby`: filter takes logical column as argument, however, we should pass string to min() or max(). PySpark has a whole class devoted to grouped data frames: `pyspark.sql.GroupedData`

`Agg`: This method lets you pass an aggregate column expression that uses any of the aggregate functions from the `pyspark.sql.functions` submodule.

`join`: `a.join(b, on='key', how='leftouter')`


### Machine Learning Pipeline
At the core of the `pyspark.ml` module are the Transformer and Estimator classes. 
`Transformer` classes have a `.transform()` method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended.
`Estimator` classes all implement a `.fit()` method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. 

Project:build a model that predicts whether or not a flight will be delayed based on the flights data 
```python
# Rename year column
planes = planes.withColumnRenamed('year', 'plane_year')

# Join the DataFrames
model_data = flights.join(planes, on='tailnum', how="leftouter")

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn("air_time", model_data.air_time.cast('integer'))
model_data = model_data.withColumn("month", model_data.month.cast('integer'))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast('integer'))

# Create the column plane_age
model_data = model_data.withColumn("plane_age",model_data.year- model_data.plane_year)

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast('integer'))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")


# One hot encoding
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")

dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")

# Make a VectorAssembler, combine all of the columns containing our features into a single column
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# pipeline
from pyspark.ml import Pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])
```


### Model Tuning and selection
Running Logistic regression and using cross validation to choose hyperparameters
```python
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the parameter grid
grid = tune.ParamGridBuilder()
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator
               )

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
```
