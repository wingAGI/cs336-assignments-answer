#!/bin/bash

NUM=500            # 要下载的数量
PARALLEL=10         # 并发数，可以适当调大，注意别太大防止被限速

mkdir -p wet

head -n "$NUM" wet.paths | \
  sed 's|^|https://data.commoncrawl.org/|' | \
  xargs -P "$PARALLEL" -n 1 -I{} wget -c -P wet {}

echo "正在解压 wet 文件夹下的所有 .gz ..."
gunzip -f wet/*.gz

echo "全部下载并解压完毕！"
