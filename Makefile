IMAGE_CU113 = samuel21119/mva2023_cu113_team1
IMAGE_CU102 = samuel21119/mva2023_cu102_team1
DATA = $(shell readlink -f data)
SHMEM_SIZE = 32G

build_cu113:
	docker build -t $(IMAGE_CU113) - < Dockerfile_cu113

build_cu102:
	docker build -t $(IMAGE_CU102) - < Dockerfile_cu102


start:
	docker run --rm	-i -t \
		-v $(PWD):/root/MVATeam1 \
		-v $(DATA):/root/MVATeam1/data \
		--gpus all \
		--shm-size $(SHMEM_SIZE) \
		$(IMAGE_CU113)
		
start_cu102:
	docker run --rm	-i -t \
		-v $(PWD):/root/MVATeam1 \
		-v $(DATA):/root/MVATeam1/data \
		--gpus all \
		--shm-size $(SHMEM_SIZE) \
		$(IMAGE_CU102)



post_install:
	cd ops_dcnv3; sh make.sh
	pip install -r requirements/sahi.txt
	pip install -v -e .
