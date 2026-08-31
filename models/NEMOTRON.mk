NEMOTRON_MODEL_NAMESPACE ?= nvidia
NEMOTRON_MODEL_VERSION ?= llama-nemotron-embed-1b-v2
NEMOTRON_ENCODE_BATCH_SIZE ?= auto
NEMOTRON_SEARCH_BATCH_SIZE ?= auto
NEMOTRON_ENCODE_STARTING_BATCH_SIZE ?= 16
NEMOTRON_SEARCH_STARTING_BATCH_SIZE ?= 16
NEMOTRON_MAX_LENGTH ?= 4096

NEMOTRON_MODEL_ID ?= $(NEMOTRON_MODEL_NAMESPACE)/$(NEMOTRON_MODEL_VERSION)
NEMOTRON_REPORT_DIR ?= $(NEMOTRON_MODEL_NAMESPACE)__$(NEMOTRON_MODEL_VERSION)
NEMOTRON_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(NEMOTRON_REPORT_DIR)/$$level)

NEMOTRON/%: reports/%/$(NEMOTRON_REPORT_DIR)
	cat reports/$*/$(NEMOTRON_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(NEMOTRON_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(NEMOTRON_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(NEMOTRON_REPORT_DIR)/LEVEL_4.txt


NEMOTRON/%/embeddings: .cache/embeddings/%/$(NEMOTRON_REPORT_DIR)/embeddings.jsonl
	@:

NEMOTRON/%/index: .cache/indexes/%/$(NEMOTRON_REPORT_DIR)/index
	@:

NEMOTRON/%/search: .cache/search_runs/%/$(NEMOTRON_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(NEMOTRON_REPORT_DIR) .cache/search_runs/%/$(NEMOTRON_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(NEMOTRON_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(NEMOTRON_REPORT_DIR)/embeddings.jsonl: corpora/%
	python -m gutenbench.encode_corpus --encoder Nemotron \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(NEMOTRON_REPORT_DIR) \
		--model-id $(NEMOTRON_MODEL_ID) \
		--batch-size $(NEMOTRON_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(NEMOTRON_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(NEMOTRON_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/$(NEMOTRON_REPORT_DIR)/index: .cache/embeddings/%/$(NEMOTRON_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(NEMOTRON_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(NEMOTRON_REPORT_DIR) \
		--output .cache/indexes/$*/$(NEMOTRON_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(NEMOTRON_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(NEMOTRON_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(NEMOTRON_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder Nemotron \
			--index .cache/indexes/$*/$(NEMOTRON_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(NEMOTRON_REPORT_DIR).$$level.trec \
			$(NEMOTRON_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(NEMOTRON_MODEL_ID) \
			--batch-size $(NEMOTRON_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(NEMOTRON_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(NEMOTRON_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/$(NEMOTRON_REPORT_DIR): .cache/indexes/%/$(NEMOTRON_REPORT_DIR)/index .cache/search_runs/%/$(NEMOTRON_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(NEMOTRON_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(NEMOTRON_REPORT_DIR).$$level.trec \
			> reports/$*/$(NEMOTRON_REPORT_DIR)/$$level.txt; \
	done
