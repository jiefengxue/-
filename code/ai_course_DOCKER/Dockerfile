
FROM docker.1ms.run/python:3.10-slim

RUN echo "deb https://mirrors.aliyun.com/debian/ trixie main non-free non-free-firmware contrib" > /etc/apt/sources.list \
    && echo "deb-src https://mirrors.aliyun.com/debian/ trixie main non-free non-free-firmware contrib" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian-security/ trixie-security main" >> /etc/apt/sources.list \
    && echo "deb-src https://mirrors.aliyun.com/debian-security/ trixie-security main" >> /etc/apt/sources.list


RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/


RUN apt-get update && apt-get install -y \
    python3-tk \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-wqy-microhei \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen zh_CN.UTF-8

# 中文环境变量
ENV LC_ALL=zh_CN.UTF-8
ENV LANG=zh_CN.UTF-8
ENV LANGUAGE=zh_CN.UTF-8

WORKDIR /app


COPY requirements.txt /app/


RUN pip install -r requirements.txt --no-cache-dir \
    && pip install mindspore==2.7.0 -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir --force-reinstall


COPY . /app/
RUN chmod +x /app/puzzle_ui.py \
    && chmod -R 755 /app/data/ \
    && chmod 644 /app/mnist_net.ckpt

# 启动命令
CMD ["python", "/app/puzzle_ui.py"]