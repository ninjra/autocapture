.PHONY: check check-gpu dev-fast dev-idle bench bench-cpu bench-gpu gate

check:
	GPU_MODE=off PROFILE=foreground poetry run black --check .
	GPU_MODE=off PROFILE=foreground poetry run ruff check .
	GPU_MODE=off PROFILE=foreground poetry run pytest -q

check-gpu:
	GPU_MODE=auto PROFILE=foreground poetry run pytest -q -m gpu

dev-fast:
	PROFILE=foreground GPU_MODE=off poetry run autocapture

dev-idle:
	PROFILE=idle GPU_MODE=auto poetry run autocapture

bench:
	PROFILE=foreground poetry run python -m autocapture.bench.run --both

bench-cpu:
	PROFILE=foreground poetry run python -m autocapture.bench.run --cpu

bench-gpu:
	PROFILE=foreground poetry run python -m autocapture.bench.run --gpu

gate: check check-gpu bench
