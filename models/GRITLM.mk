GRITLM_MODEL_NAMESPACE ?= GritLM
GRITLM_MODEL_VERSION ?= GritLM-7B
GRITLM_ENCODE_BATCH_SIZE ?= auto
GRITLM_SEARCH_BATCH_SIZE ?= auto
GRITLM_ENCODE_STARTING_BATCH_SIZE ?= 16
GRITLM_SEARCH_STARTING_BATCH_SIZE ?= 16
GRITLM_MAX_LENGTH ?= 4096
GRITLM_TORCH_DTYPE ?= bfloat16

GRITLM_MODEL_ID ?= $(GRITLM_MODEL_NAMESPACE)/$(GRITLM_MODEL_VERSION)
GRITLM_REPORT_DIR ?= $(GRITLM_MODEL_NAMESPACE)__$(GRITLM_MODEL_VERSION)
GRITLM_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(GRITLM_REPORT_DIR)/$$level)

GRITLM/%: reports/%/$(GRITLM_REPORT_DIR)
	cat reports/$*/$(GRITLM_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(GRITLM_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(GRITLM_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(GRITLM_REPORT_DIR)/LEVEL_4.txt


GRITLM/%/embeddings: .cache/embeddings/%/$(GRITLM_REPORT_DIR)/embeddings.jsonl
	@:

GRITLM/%/index: .cache/indexes/%/$(GRITLM_REPORT_DIR)/index
	@:

GRITLM/%/search: .cache/search_runs/%/$(GRITLM_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(GRITLM_REPORT_DIR) .cache/search_runs/%/$(GRITLM_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(GRITLM_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(GRITLM_REPORT_DIR)/embeddings.jsonl: corpora/%.csv corpora/%
	python -m gutenbench.encode_corpus --encoder GritLM \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(GRITLM_REPORT_DIR) \
		--model-id $(GRITLM_MODEL_ID) \
		--batch-size $(GRITLM_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(GRITLM_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(GRITLM_MAX_LENGTH) \
		--torch-dtype $(GRITLM_TORCH_DTYPE) \
		$(DEVICE_ARG)

.cache/indexes/%/$(GRITLM_REPORT_DIR)/index: .cache/embeddings/%/$(GRITLM_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(GRITLM_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(GRITLM_REPORT_DIR) \
		--output .cache/indexes/$*/$(GRITLM_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(GRITLM_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(GRITLM_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(GRITLM_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder GritLM \
			--index .cache/indexes/$*/$(GRITLM_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(GRITLM_REPORT_DIR).$$level.trec \
			$(GRITLM_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(GRITLM_MODEL_ID) \
			--batch-size $(GRITLM_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(GRITLM_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(GRITLM_MAX_LENGTH) \
			--torch-dtype $(GRITLM_TORCH_DTYPE) \
			$(DEVICE_ARG); \
	done

reports/%/$(GRITLM_REPORT_DIR): .cache/indexes/%/$(GRITLM_REPORT_DIR)/index .cache/search_runs/%/$(GRITLM_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(GRITLM_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(GRITLM_REPORT_DIR).$$level.trec \
			> reports/$*/$(GRITLM_REPORT_DIR)/$$level.txt; \
	done
