KALM_MODEL_NAMESPACE ?= tencent
KALM_MODEL_VERSION ?= KaLM-Embedding-Gemma3-12B-2511
KALM_ENCODE_BATCH_SIZE ?= auto
KALM_SEARCH_BATCH_SIZE ?= auto
KALM_ENCODE_STARTING_BATCH_SIZE ?= 8
KALM_SEARCH_STARTING_BATCH_SIZE ?= 8
KALM_MAX_LENGTH ?= 4096

KALM_MODEL_ID ?= $(KALM_MODEL_NAMESPACE)/$(KALM_MODEL_VERSION)
KALM_REPORT_DIR ?= $(KALM_MODEL_NAMESPACE)__$(KALM_MODEL_VERSION)
KALM_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(KALM_REPORT_DIR)/$$level)

KALM/%: reports/%/$(KALM_REPORT_DIR)
	cat reports/$*/$(KALM_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(KALM_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(KALM_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(KALM_REPORT_DIR)/LEVEL_4.txt


KALM/%/embeddings: .cache/embeddings/%/$(KALM_REPORT_DIR)/embeddings.jsonl
	@:

KALM/%/index: .cache/indexes/%/$(KALM_REPORT_DIR)/index
	@:

KALM/%/search: .cache/search_runs/%/$(KALM_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(KALM_REPORT_DIR) .cache/search_runs/%/$(KALM_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(KALM_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(KALM_REPORT_DIR)/embeddings.jsonl: corpora/%
	python -m gutenbench.encode_corpus --encoder KaLM \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(KALM_REPORT_DIR) \
		--model-id $(KALM_MODEL_ID) \
		--batch-size $(KALM_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(KALM_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(KALM_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/$(KALM_REPORT_DIR)/index: .cache/embeddings/%/$(KALM_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(KALM_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(KALM_REPORT_DIR) \
		--output .cache/indexes/$*/$(KALM_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(KALM_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(KALM_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(KALM_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder KaLM \
			--index .cache/indexes/$*/$(KALM_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(KALM_REPORT_DIR).$$level.trec \
			$(KALM_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(KALM_MODEL_ID) \
			--batch-size $(KALM_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(KALM_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(KALM_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/$(KALM_REPORT_DIR): .cache/indexes/%/$(KALM_REPORT_DIR)/index .cache/search_runs/%/$(KALM_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(KALM_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(KALM_REPORT_DIR).$$level.trec \
			> reports/$*/$(KALM_REPORT_DIR)/$$level.txt; \
	done
