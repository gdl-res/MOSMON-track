# Manuscript figure generation.
#
# Inputs are resolved from configs/figures.yaml — point `paths.mosmon_outputs`
# at the tracking pipeline's output tree before running. Every target is
# non-interactive and rewrites its outputs in figures/generated/.

PYTHON ?= python
FIGURE_DIR := scripts/figures

.PHONY: figures fig03 fig04 fig05 fig06 fig07 \
        test-figures test-benchmark clean-figures

## Regenerate every manuscript figure (PDF + PNG + metadata + data CSV + panels)
figures: fig03 fig04 fig05 fig06 fig07

## Figure 3 — tracking quality across crowding
fig03:
	$(PYTHON) $(FIGURE_DIR)/fig03_tracking_crowding.py

## Figure 4 — population-level trajectory information
fig04:
	$(PYTHON) $(FIGURE_DIR)/fig04_population_analysis.py

## Figure 5 — controlled dual-container analysis
fig05:
	$(PYTHON) $(FIGURE_DIR)/fig05_dual_container.py

## Figure 6 — fixed-detector multi-tracker benchmark
fig06:
	$(PYTHON) $(FIGURE_DIR)/fig06_tracker_benchmark.py

## Figure 7 — how far tracker choice reaches into the biology
fig07:
	$(PYTHON) $(FIGURE_DIR)/fig07_tracker_downstream.py

## Unit tests for the figure helpers (no outputs volume or model weights needed)
test-figures:
	$(PYTHON) -m pytest tests/test_figures.py -q

## Unit tests for the tracker benchmark (no weights, no videos, no GPU needed)
test-benchmark:
	$(PYTHON) -m pytest tests/test_tracker_adapters.py tests/test_benchmark_analysis.py -q

## Remove generated figures (source scripts and config are untouched)
clean-figures:
	rm -rf figures/generated
