INSTRUCTOR_MODEL_NAMESPACE ?= hkunlp
INSTRUCTOR_MODEL_VERSION ?= instructor-base
INSTRUCTOR_ENCODE_BATCH_SIZE ?= auto
INSTRUCTOR_SEARCH_BATCH_SIZE ?= auto
INSTRUCTOR_ENCODE_STARTING_BATCH_SIZE ?= 32
INSTRUCTOR_SEARCH_STARTING_BATCH_SIZE ?= 32
INSTRUCTOR_MAX_LENGTH ?= 512

INSTRUCTOR_MODEL_ID ?= $(INSTRUCTOR_MODEL_NAMESPACE)/$(INSTRUCTOR_MODEL_VERSION)
INSTRUCTOR_REPORT_DIR ?= $(INSTRUCTOR_MODEL_NAMESPACE)__$(INSTRUCTOR_MODEL_VERSION)
INSTRUCTOR_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/$(INSTRUCTOR_REPORT_DIR)/$$level)

INSTRUCTOR/%: reports/%/$(INSTRUCTOR_REPORT_DIR)
	cat reports/$*/$(INSTRUCTOR_REPORT_DIR)/LEVEL_1.txt
	cat reports/$*/$(INSTRUCTOR_REPORT_DIR)/LEVEL_2.txt
	cat reports/$*/$(INSTRUCTOR_REPORT_DIR)/LEVEL_3.txt
	cat reports/$*/$(INSTRUCTOR_REPORT_DIR)/LEVEL_4.txt


INSTRUCTOR/%/embeddings: .cache/embeddings/%/$(INSTRUCTOR_REPORT_DIR)/embeddings.jsonl
	@:

INSTRUCTOR/%/index: .cache/indexes/%/$(INSTRUCTOR_REPORT_DIR)/index
	@:

INSTRUCTOR/%/search: .cache/search_runs/%/$(INSTRUCTOR_REPORT_DIR).LEVEL_1.trec
	@:

.PRECIOUS: reports/%/$(INSTRUCTOR_REPORT_DIR) .cache/search_runs/%/$(INSTRUCTOR_REPORT_DIR).LEVEL_1.trec .cache/embeddings/%/$(INSTRUCTOR_REPORT_DIR)/embeddings.jsonl

.cache/embeddings/%/$(INSTRUCTOR_REPORT_DIR)/embeddings.jsonl: corpora/%.csv corpora/%
	python -m gutenbench.encode_corpus --encoder InstructOR \
		--input corpora/$* \
		--output .cache/embeddings/$*/$(INSTRUCTOR_REPORT_DIR) \
		--model-id $(INSTRUCTOR_MODEL_ID) \
		--batch-size $(INSTRUCTOR_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(INSTRUCTOR_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(INSTRUCTOR_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/$(INSTRUCTOR_REPORT_DIR)/index: .cache/embeddings/%/$(INSTRUCTOR_REPORT_DIR)/embeddings.jsonl
	mkdir -p .cache/indexes/$*/$(INSTRUCTOR_REPORT_DIR)
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/$(INSTRUCTOR_REPORT_DIR) \
		--output .cache/indexes/$*/$(INSTRUCTOR_REPORT_DIR) \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/$(INSTRUCTOR_REPORT_DIR)/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/$(INSTRUCTOR_REPORT_DIR).LEVEL_1.trec: .cache/indexes/%/$(INSTRUCTOR_REPORT_DIR)/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder InstructOR \
			--index .cache/indexes/$*/$(INSTRUCTOR_REPORT_DIR) \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/$(INSTRUCTOR_REPORT_DIR).$$level.trec \
			$(INSTRUCTOR_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(INSTRUCTOR_MODEL_ID) \
			--batch-size $(INSTRUCTOR_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(INSTRUCTOR_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(INSTRUCTOR_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/$(INSTRUCTOR_REPORT_DIR): .cache/indexes/%/$(INSTRUCTOR_REPORT_DIR)/index .cache/search_runs/%/$(INSTRUCTOR_REPORT_DIR).LEVEL_1.trec
	mkdir -p reports/$*/$(INSTRUCTOR_REPORT_DIR)
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/$(INSTRUCTOR_REPORT_DIR).$$level.trec \
			> reports/$*/$(INSTRUCTOR_REPORT_DIR)/$$level.txt; \
	done
