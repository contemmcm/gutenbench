QWEN_MODEL_NAMESPACE ?= Qwen
QWEN_MODEL_VERSION ?= Qwen3-Embedding-0.6B
QWEN_ENCODE_BATCH_SIZE ?= auto
QWEN_SEARCH_BATCH_SIZE ?= auto
QWEN_ENCODE_STARTING_BATCH_SIZE ?= 16
QWEN_SEARCH_STARTING_BATCH_SIZE ?= 16
QWEN_MAX_LENGTH ?= 4096

QWEN_MODEL_ID ?= $(QWEN_MODEL_NAMESPACE)/$(QWEN_MODEL_VERSION)
QWEN_REPORT_DIR ?= $(QWEN_MODEL_NAMESPACE)__$(QWEN_MODEL_VERSION)
QWEN_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(QWEN_REPORT_DIR)/$$level)

QWEN/%: reports/%/$(QWEN_REPORT_DIR)
	cat reports/$*/$(QWEN_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(QWEN_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(QWEN_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(QWEN_REPORT_DIR)/LEVEL_4.txt

QWEN/%/embeddings: .cache/embeddings/%/$(QWEN_REPORT_DIR)/embeddings.jsonl
	@:

QWEN/%/index: .cache/indexes/%/$(QWEN_REPORT_DIR)/index
	@:

QWEN/%/search: .cache/search_runs/%/$(QWEN_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(QWEN_REPORT_DIR) .cache/search_runs/%/$(QWEN_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(QWEN_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(QWEN_REPORT_DIR)/embeddings.jsonl: corpora/%
	python -m gutenbench.encode_corpus --encoder Qwen3 \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(QWEN_REPORT_DIR) \
		--model-id $(QWEN_MODEL_ID) \
		--batch-size $(QWEN_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(QWEN_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(QWEN_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/$(QWEN_REPORT_DIR)/index: .cache/embeddings/%/$(QWEN_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(QWEN_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(QWEN_REPORT_DIR) \
		--output .cache/indexes/$*/$(QWEN_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(QWEN_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(QWEN_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(QWEN_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder Qwen3 \
			--index .cache/indexes/$*/$(QWEN_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(QWEN_REPORT_DIR).$$level.trec \
			$(QWEN_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(QWEN_MODEL_ID) \
			--batch-size $(QWEN_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(QWEN_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(QWEN_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/$(QWEN_REPORT_DIR): .cache/indexes/%/$(QWEN_REPORT_DIR)/index .cache/search_runs/%/$(QWEN_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(QWEN_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(QWEN_REPORT_DIR).$$level.trec \
			> reports/$*/$(QWEN_REPORT_DIR)/$$level.txt; \
	done
