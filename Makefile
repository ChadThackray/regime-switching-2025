.PHONY:  lint


lint:
	uv run ruff check .
	uv run ruff format .
	uv run ruff check --fix .
