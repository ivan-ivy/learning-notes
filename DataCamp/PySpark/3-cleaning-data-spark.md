## Cleaning Data with PySpark

### Data Frame Details
Creating a defined schema helps with data quality and import performance.

```python
# Import the pyspark.sql.types library
from pyspark.sql.types import * 

# Define a new schema using the StructType method
# all fields here are not nullable
people_schema = StructType([
  # Define a StructField for each field
  StructField('name', StringType(), False),
  StructField('age', IntegerType(), False),
  StructField('city', StringType(), False)
])

```

* Immutability and lazy processing

Spark data frame is immutable, unable to be directly modified.
easier for sharing files between different cluster

lazy processing: very little happens until an action is performed

```python
# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')

# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))

# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])

# Show the DataFrame
aa_dfw_df.show()

```

CSV file is not optimal format for spark.
Difficulties: 
No defined schema
Nested data requires special handling
Encoding format limited

Slow to parse
Files cannot be ltered (no "predicate pushdown")
Any intermediate use requires redening schema

Parquet is a better format:
A columnar data format
Supported in Spark and other data processing frameworks
Supports predicate pushdown
Automatically stores schema information

```python
# read
df = spark.read.format('parquet').load('filename.parquet')
df = spark.read.parquet('filename.parquet')
# write
df.write.format('parquet').save('filename.parquet')
df.write.parquet('filename.parquet')

# sql
flight_df = spark.read.parquet('flights.parquet')
flight_df.createOrReplaceTempView('flights')
short_flights_df = spark.sql('SELECT * FROM flights WHERE flightduration < 100')
```

### Dataframe operations

* filter/ where
* select
* withColumn
* drop
  
```python
voter_df.filter(voter_df.name.like('M%'))
voter_df.filter(voter_df['name'].isNotNull())
voter_df.filter(voter_df.date.year > 1800)
voter_df.where(voter_df['_c0'].contains('VOTE'))
voter_df.where(~ voter_df._c1.isNull()) # ~ can make opposite result
# string transformation
import pyspark.sql.functions as F
voter_df.withColumn('upper', F.upper('name'))
voter_df.withColumn('splits', F.split('name',' '))
voter_df.withColumn('year', voter_df['_c4'].cast(IntegerType()))
# 


# 
```
ArrayType functions
`.size(<column>)` - returns length of arrayType() column
`.getItem(<index>)` - used to retrieve a specic item at index oflist column.

```python
# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn('splits', F.split(voter_df.VOTER_NAME, '\s+'))

# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn('first_name', voter_df.splits.getItem(0))

# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn('last_name', voter_df.splits.getItem(F.size('splits') - 1))

# Drop the splits column
voter_df = voter_df.drop('splits')

# Show the voter_df DataFrame
voter_df.show()
```

when/ otherwise
```python
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand())
                               .when(voter_df.TITLE == 'Mayor', 2)
                               .otherwise(0))

# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val==0).show()
```

UDF
python method, wrapped via the `pyspark.sql.functions.udf`

Partitioning and lazy processing

ID 
id actually is not very parallel
`pyspark.sql.functions.monotonically_increasing_id()`
Integer (64-bit), increases in value, unique
Not necessarily sequential (gaps exist)
Completely parallel

```python
# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()

# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
```


Caching a DataFrame
 caching can improve performance when reusing DataFrames
```python
voter_df = voter_df.cache()
voter_df.is_cached
voter_df.unpersist() # removed from cache
```