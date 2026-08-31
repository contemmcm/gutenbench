INFX_MODEL_NAMESPACE ?= infly
INFX_MODEL_VERSION ?= inf-retriever-v1-1.5b
INFX_ENCODE_BATCH_SIZE ?= auto
INFX_SEARCH_BATCH_SIZE ?= auto
INFX_ENCODE_STARTING_BATCH_SIZE ?= 16
INFX_SEARCH_STARTING_BATCH_SIZE ?= 16
INFX_MAX_LENGTH ?= 4096
INFX_TORCH_DTYPE ?= bfloat16

INFX_MODEL_ID ?= $(INFX_MODEL_NAMESPACE)/$(INFX_MODEL_VERSION)
INFX_REPORT_DIR ?= $(INFX_MODEL_NAMESPACE)__$(INFX_MODEL_VERSION)
INFX_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(INFX_REPORT_DIR)/$$level)

INFX/%: reports/%/$(INFX_REPORT_DIR)
	cat reports/$*/$(INFX_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(INFX_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(INFX_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(INFX_REPORT_DIR)/LEVEL_4.txt


INFX/%/embeddings: .cache/embeddings/%/$(INFX_REPORT_DIR)/embeddings.jsonl
	@:

INFX/%/index: .cache/indexes/%/$(INFX_REPORT_DIR)/index
	@:

INFX/%/search: .cache/search_runs/%/$(INFX_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(INFX_REPORT_DIR) .cache/search_runs/%/$(INFX_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(INFX_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(INFX_REPORT_DIR)/embeddings.jsonl: corpora/%.csv corpora/%
	python -m gutenbench.encode_corpus --encoder InfX \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(INFX_REPORT_DIR) \
		--model-id $(INFX_MODEL_ID) \
		--batch-size $(INFX_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(INFX_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(INFX_MAX_LENGTH) \
		--torch-dtype $(INFX_TORCH_DTYPE) \
		$(DEVICE_ARG)

.cache/indexes/%/$(INFX_REPORT_DIR)/index: .cache/embeddings/%/$(INFX_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(INFX_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(INFX_REPORT_DIR) \
		--output .cache/indexes/$*/$(INFX_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(INFX_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(INFX_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(INFX_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder InfX \
			--index .cache/indexes/$*/$(INFX_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(INFX_REPORT_DIR).$$level.trec \
			$(INFX_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(INFX_MODEL_ID) \
			--batch-size $(INFX_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(INFX_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(INFX_MAX_LENGTH) \
			--torch-dtype $(INFX_TORCH_DTYPE) \
			$(DEVICE_ARG); \
	done

reports/%/$(INFX_REPORT_DIR): .cache/indexes/%/$(INFX_REPORT_DIR)/index .cache/search_runs/%/$(INFX_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(INFX_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(INFX_REPORT_DIR).$$level.trec \
			> reports/$*/$(INFX_REPORT_DIR)/$$level.txt; \
	done
