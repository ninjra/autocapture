MODE ?= both

.PHONY: check check-gpu dev-fast dev-idle bench bench-cpu bench-gpu gate

check:
	AUTOCAPTURE_GPU_MODE=off AUTOCAPTURE_PROFILE=foreground poetry run black --check .
	AUTOCAPTURE_GPU_MODE=off AUTOCAPTURE_PROFILE=foreground poetry run ruff check .
	AUTOCAPTURE_GPU_MODE=off AUTOCAPTURE_PROFILE=foreground poetry run pytest -q -m "not gpu"

check-gpu:
	AUTOCAPTURE_GPU_MODE=auto AUTOCAPTURE_PROFILE=foreground poetry run pytest -q -m gpu

dev-fast:
	AUTOCAPTURE_PROFILE=foreground AUTOCAPTURE_GPU_MODE=off \
		AUTOCAPTURE_FOREGROUND_MAX_WORKERS=2 AUTOCAPTURE_FOREGROUND_BATCH_SIZE=4 \
		poetry run autocapture

dev-idle:
	AUTOCAPTURE_PROFILE=idle AUTOCAPTURE_GPU_MODE=auto poetry run autocapture

bench:
	AUTOCAPTURE_PROFILE=foreground poetry run python -m autocapture.bench.run --mode $(MODE)

bench-cpu:
	AUTOCAPTURE_PROFILE=foreground poetry run python -m autocapture.bench.run --mode cpu

bench-gpu:
	AUTOCAPTURE_PROFILE=foreground poetry run python -m autocapture.bench.run --mode gpu

gate: check check-gpu bench
