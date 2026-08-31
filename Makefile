PYTORCH_DEVICE ?=

DEVICE_ARG = $(if $(PYTORCH_DEVICE),--device $(PYTORCH_DEVICE),)
LEVELS ?= LEVEL_1 LEVEL_2 LEVEL_3 LEVEL_4
ALL_LEVELS = $(LEVELS) $(if $(SEARCH_CANDIDATES),CANDIDATES)

.DELETE_ON_ERROR:
.SECONDARY: corpora/%.csv
.PRECIOUS: .cache/dataset/%

corpora/%.csv: resources/headings.csv
	python scripts/chunk.py --index resources/headings.csv --id $*

corpora/%: corpora/%.csv
	python scripts/dump.py --id $*
	touch $@

clean:
	rm -rf corpora .cache reports

include models/BM25.mk
include models/BGE.mk
include models/QWEN.mk
include models/NEMOTRON.mk
include models/KALM.mk
include models/SBERT.mk
include models/INSTRUCTOR.mk
include models/INFX.mk
include models/GRITLM.mk

