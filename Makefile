WORKDIRNAME = $(shell sh -c "pwd | tr '/' '\\n' | tail -n 1 | tr A-Z a-z")
EXAMPLES = /opt/uut/examples

DOCKERRUN = docker run -ti ${WORKDIRNAME}_test_iz_estimation 

PROGRESS = --progress plain


.PHONY: pip_build pip_build_no_cache

pip_build: 
	docker-compose build ${PROGRESS} test_iz_estimation 

pip_build_no_cache: 
	docker-compose build ${PROGRESS} --no-cache test_iz_estimation && ${DOCKERRUN} sh

test: pip_build
	${DOCKERRUN} sh -c "cd ./test/ && pytest integration.py"

container_shell: pip_build
	${DOCKERRUN} sh
