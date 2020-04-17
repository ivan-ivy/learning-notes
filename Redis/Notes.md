# Chapter 1
Redis是REmote DIctionary Server的缩写。
## 特性
### 存储结构
以字典结构存储数据。Redis中键值可以使字符串、散列、列表、集合、有序集合等。
相比MySQL的优势是Redis支持多种数据类型，可以直接将数据映射到Redis中，以及Redis为不同数据类型提供了很多方便的操作。
### 内存存储、持久化
Redis中所有数据都存在内存中。Redis提供持久化支持，可以将内存中数据异步存储到硬盘中，同时不影响继续提供服务。
### 功能丰富
可以作为换成、队列系统等。可以为每个键设置Time to Live，到期键自动删除。
### 简单
```
HGET post:1 title
```
读取键名为post：1的散列类型键的tile字段的值