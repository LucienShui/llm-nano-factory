FROM nvcr.io/nvidia/pytorch:24.01-py3
COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN echo -e 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse' > /etc/apt/sources.list && apt update && apt install openssh-server -y && usermod --password $(echo root | openssl passwd -1 -stdin) root && echo -e 'Port 10022\nPermitRootLogin yes' > /etc/ssh/sshd_config.d/user.conf && ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys
