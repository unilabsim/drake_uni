# Common development commands for DrakeUni.
#
# Drake itself is intentionally not pulled from PyPI: the native extension is
# compiled against a local Drake C++ installation.  Set DRAKE_HOME to that
# installation prefix for the build targets below.

.PHONY: sync
sync:
	uv sync

.PHONY: install
install:
	uv pip install --force-reinstall --no-deps --no-build-isolation -e .

# Build the optional pybind11 extension against the Drake installation in
# DRAKE_HOME (for example, make build DRAKE_HOME=/opt/drake).
.PHONY: build
build:
	@test -n "$(DRAKE_HOME)" || (echo "usage: make build DRAKE_HOME=/path/to/drake/install" && exit 1)
	uv run python scripts/build_drake_batch.py --drake-home "$(DRAKE_HOME)"

.PHONY: build-dry-run
build-dry-run:
	@test -n "$(DRAKE_HOME)" || (echo "usage: make build-dry-run DRAKE_HOME=/path/to/drake/install" && exit 1)
	uv run python scripts/build_drake_batch.py --drake-home "$(DRAKE_HOME)" --dry-run

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: test
test:
	uv run pytest -q

# Keep the already-built native extension and current environment untouched.
.PHONY: test-no-sync
test-no-sync:
	uv run --no-sync pytest -q

.PHONY: check
check: lint test

.PHONY: package
package:
	uv build --sdist

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist
