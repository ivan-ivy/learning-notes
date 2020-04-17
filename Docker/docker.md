# Docker Notes

## 安装

### Windows
1. 开启Hyper-V
2. 下载安装docker desktop

### Mac
直接下载docker desktop

## 容器使用
### Hello World
```shell
docker run ubuntu:15.10 /bin/eco "Hello world"
```
ubuntu:15.10: 指定要运行的镜像，如果本地主机上不存在，会从docker hub上下载公共镜像

/bin/eco "Hello world": 在启动的容器里执行的命令

```shell
docker pull ubuntu
```
如果本地没有ubuntu镜像，可以通过pull来载入

### 常用命令
**交互式：**
```shell
docker run -i -t ubuntu:15.10 /bin/bash
```
-t: 在新容器内指定一个伪终端或终端。
-i: 允许你对容器内的标准输入 (STDIN) 进行交互。
运行之后会进入容器的命令行，使用ctrl+D来退出，或者输入exit。

**后台模式：**
```shell
docker run -d ubuntu:15.10 /bin/sh -c "while true; do echo hello world; sleep 1; done"
```
-d：容器在后台运行。

`docker ps`: 查看当前运行的容器。
`docker logs + id`: 查看容器内的标准输出。-f: 让 docker logs 像使用 tail -f 一样来输出容器内部的标准输出。
`docker stop + id`: 停止当前运行的容器。
`docker start + id`: 启动已经停止的容器。
`docker restart + id`: 重启的容器。
`docker inspect`: 查看 Docker 的底层信息。它会返回一个 JSON 文件记录着 Docker 容器的配置和状态信息。

**进入容器：**
`docker attach`

`docker exec`：推荐使用 `docker exec` 命令，因为此退出容器终端，不会导致容器的停止。

**导入导出：**
`docker export 1e560fca3906 > ubuntu.tar`

`cat docker/ubuntu.tar | docker import - test/ubuntu:v1`


**删除容器：** ` docker rm -f 1e560fca3906` 删除时需要容器处于停滞状态。

**运行web应用：** 
```shell
docker pull training/webapp
docker run -d -P training/webapp python app.py
```
-P:将容器内部使用的网络端口映射到我们使用的主机上。
-p 5000:5000 : 将容器内部的5000端口映射到本地主机的5000端口。
使用docker ps查看状态时，在port一栏会显示端口映射关系。也可以使用`docker port`来查看对应的端口。


## 镜像使用
`docker images`: 查看本地主机上镜像
`docker pull`: 获取新的镜像
`docker search`: 按照关键词来搜寻镜像
`docker build`: 根据dockerfile来创建镜像，e.g.
`docker tag`: 为镜像添加一个新的标签
```
~$ cat Dockerfile 
FROM    centos:6.7
MAINTAINER      Fisher "fisher@sudops.com"

RUN     /bin/echo 'root:123456' |chpasswd
RUN     useradd runoob
RUN     /bin/echo 'runoob:123456' |chpasswd
RUN     /bin/echo -e "LANG=\"en_US.UTF-8\"" >/etc/default/local
EXPOSE  22
EXPOSE  80
CMD     /usr/sbin/sshd -D
```

FROM：定制的镜像都是基于 FROM 的镜像
RUN：用于执行后面跟着的命令行命令。
Dockerfile 的指令每执行一次都会在 docker 上新建一层。所以过多无意义的层，会造成镜像膨胀过大。以 && 符号连接命令，这样执行后，只会创建 1 层镜像。
CMD: 类似于 RUN 指令，用于运行程序，但二者运行的时间点不同:CMD 在docker run 时运行。RUN 是在 docker build。
ENV: 设置环境变量，定义了环境变量，那么在后续的指令中，就可以使用这个环境变量。
VOLUME: 定义匿名数据卷。在启动容器时忘记挂载数据卷，会自动挂载到匿名卷。避免重要的数据，因容器重启而丢失，这是非常致命的。避免容器不断变大。
EXPOSE: 仅仅只是声明端口。帮助镜像使用者理解这个镜像服务的守护端口，以方便配置映射。在运行时使用随机端口映射时，也就是 docker run -P 时，会自动随机映射 EXPOSE 的端口。
WORKDIR: 指定工作目录。用 WORKDIR 指定的工作目录，会在构建镜像的每一层中都存在。（WORKDIR 指定的工作目录，必须是提前创建好的）。


## Redis
```
docker pull redis:latest
docker run -itd --name redis-test -p 6379:6379 redis

root@d7033b9784ee:/data# redis-cli
127.0.0.1:6379> set test 1
OK
```