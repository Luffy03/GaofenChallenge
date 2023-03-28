# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential \
 cmake \
 curl \
 ca-certificates \
 libjpeg-dev \
 libpng-dev && \
 rm -rf /var/lib/apt/lists/*
RUN pip install ttach
RUN pip install yacs
RUN pip install timm==0.3.2
# 将程序复制容器内，表示在/workspace 路径下
COPY workspace /workspace
# 确定容器启动时程序运行路径
WORKDIR /workspace
# 确定容器启动命令。以 python 示例，python 表示编译器，run.py 表示执
# 行文件，/input_path 和/output_path 为容器内绝对路径，测评时会自动将
# 测试数据挂载到容器内/input_path 路径，无需修改
CMD ["python", "run.py", "/input_path", "/output_path"]
