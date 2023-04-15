IMAGE = samuel21119/mva2023_cu115_team1
DATA = $(shell readlink -f data)

build:
	docker build -t $(IMAGE) .

start:
	docker run --rm	-i -t \
		-v $(PWD):/root/MVATeam1 \
		-v $(DATA):/root/MVATeam1/data \
		--gpus all \
		$(IMAGE)

post_install:
	cd ops_dcnv3; sh make.sh
	pip install -v -e .