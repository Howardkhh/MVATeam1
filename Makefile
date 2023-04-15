IMAGE = samuel21119/mva2023_cu115_team1

build:
	docker build -t $(IMAGE) .


start:
	docker run --rm	-i -t \
		--mount type=bind,source=$(PWD),target=/root/MVATeam1 \
		--gpus all \
		$(IMAGE)

post_install:
	cd ops_dcnv3; sh make.sh
	pip install -v -e .