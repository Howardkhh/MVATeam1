IMAGE = samuel21119/mva2023_cu113_team1
DATA = $(shell readlink -f data)
SHMEM_SIZE = 32G

build:
	docker build -t $(IMAGE) - < Dockerfile
start:
	docker run --rm	-i -t \
		-v $(PWD):/root/MVATeam1 \
		-v $(DATA):/root/MVATeam1/data \
		--gpus all \
		--shm-size $(SHMEM_SIZE) \
		$(IMAGE)

post_install:
	cd ops_dcnv3; sh make.sh
	pip install -r requirements/sahi.txt
	pip install -v -e .
