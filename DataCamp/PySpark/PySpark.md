# DataCamp: PySpark

## Intro to Spark
Spark is a platform for cluster computing. Spark lets you spread data and computations over clusters with multiple nodes.

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

```
Spark DataFrame is immutable. To update, just reassign the returned DataFrame.

