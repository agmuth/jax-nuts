install:
	poetry install

format:
	poetry run black nuts/
	poetry run black tests/

	poetry run isort nuts/
	poetry run isort tests/

lint:
	poetry run ruff --fix nuts/
	poetry run ruff --fix tests/

test:
	poetry run pytest --cov=tests tests/