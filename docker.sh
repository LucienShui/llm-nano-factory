docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=4,5"' \
	-v "${PWD}:/pwd" \
	-p 8001:8001 \
	llm-nano-factory-runtime:latest \
	bash -c "echo -e 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse\\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse' > /etc/apt/sources.list && apt update && apt install openssh-server -y && usermod --password \$(echo root | openssl passwd -1 -stdin) root && echo -e 'PermitRootLogin yes' > /etc/ssh/sshd_config.d/user.conf && service ssh start && tail -f /dev/null"
