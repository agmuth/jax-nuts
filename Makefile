install:
	poetry install

format:
	poetry run black jaxnuts/
	poetry run black tests/

	poetry run isort jaxnuts/
	poetry run isort tests/

lint:
	poetry run ruff --fix jaxnuts/
	poetry run ruff --fix tests/

test:
	poetry run pytest --cov=jaxnuts tests/