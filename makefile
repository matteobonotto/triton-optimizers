
install:
	pip install poetry
	poetry install

test:
	poetry run pytest -m "not slow" -vs

style:
	poetry run black src/triton_optimizers

benchmark:
	python -m triton_optimizers.benckmarck #--all

