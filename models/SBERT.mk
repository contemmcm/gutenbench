SBERT_MODEL_NAMESPACE ?= sentence-transformers
SBERT_MODEL_VERSION ?= all-mpnet-base-v2
SBERT_ENCODE_BATCH_SIZE ?= auto
SBERT_SEARCH_BATCH_SIZE ?= auto
SBERT_ENCODE_STARTING_BATCH_SIZE ?= 64
SBERT_SEARCH_STARTING_BATCH_SIZE ?= 64
SBERT_MAX_LENGTH ?= 512

SBERT_MODEL_ID ?= $(SBERT_MODEL_NAMESPACE)/$(SBERT_MODEL_VERSION)
SBERT_REPORT_DIR ?= $(SBERT_MODEL_NAMESPACE)__$(SBERT_MODEL_VERSION)
SBERT_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(SBERT_REPORT_DIR)/$$level)

SBERT/%: reports/%/$(SBERT_REPORT_DIR)
	cat reports/$*/$(SBERT_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(SBERT_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(SBERT_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(SBERT_REPORT_DIR)/LEVEL_4.txt


SBERT/%/embeddings: .cache/embeddings/%/$(SBERT_REPORT_DIR)/embeddings.jsonl
	@:

SBERT/%/index: .cache/indexes/%/$(SBERT_REPORT_DIR)/index
	@:

SBERT/%/search: .cache/search_runs/%/$(SBERT_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(SBERT_REPORT_DIR) .cache/search_runs/%/$(SBERT_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(SBERT_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(SBERT_REPORT_DIR)/embeddings.jsonl: corpora/%.csv corpora/%
	python -m gutenbench.encode_corpus --encoder SBERT \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(SBERT_REPORT_DIR) \
		--model-id $(SBERT_MODEL_ID) \
		--batch-size $(SBERT_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(SBERT_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(SBERT_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/$(SBERT_REPORT_DIR)/index: .cache/embeddings/%/$(SBERT_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(SBERT_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(SBERT_REPORT_DIR) \
		--output .cache/indexes/$*/$(SBERT_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(SBERT_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(SBERT_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(SBERT_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder SBERT \
			--index .cache/indexes/$*/$(SBERT_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(SBERT_REPORT_DIR).$$level.trec \
			$(SBERT_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(SBERT_MODEL_ID) \
			--batch-size $(SBERT_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(SBERT_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(SBERT_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/$(SBERT_REPORT_DIR): .cache/indexes/%/$(SBERT_REPORT_DIR)/index .cache/search_runs/%/$(SBERT_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(SBERT_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(SBERT_REPORT_DIR).$$level.trec \
			> reports/$*/$(SBERT_REPORT_DIR)/$$level.txt; \
	done
