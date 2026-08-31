BGE_ENCODE_BATCH_SIZE ?= auto
BGE_SEARCH_BATCH_SIZE ?= auto
BGE_ENCODE_STARTING_BATCH_SIZE ?= 64
BGE_SEARCH_STARTING_BATCH_SIZE ?= 64
BGE_MAX_LENGTH ?= 512
BGE_MODEL_ID ?= BAAI/bge-large-en-v1.5
BGE_QUERY_CACHE_ARG = $(if $(SAVE_QUERY_EMBEDDINGS),--queries-cache .cache/dataset/BGE/$$level)

BGE/%: reports/%/BGE
	cat reports/$*/BGE/LEVEL_1.txt
	cat reports/$*/BGE/LEVEL_2.txt
	cat reports/$*/BGE/LEVEL_3.txt
	cat reports/$*/BGE/LEVEL_4.txt

BGE/%/embeddings: .cache/embeddings/%/BGE/embeddings.jsonl
	@:

BGE/%/index: .cache/indexes/%/BGE/index
	@:

BGE/%/search: .cache/search_runs/%/BGE.LEVEL_1.trec
	@:

.PRECIOUS: reports/%/BGE .cache/search_runs/%/BGE.LEVEL_1.trec .cache/embeddings/%/BGE/embeddings.jsonl

.cache/embeddings/%/BGE/embeddings.jsonl: corpora/%.csv corpora/%
	python -m gutenbench.encode_corpus --encoder BGE \
		--input corpora/$* \
		--output .cache/embeddings/$*/BGE \
		--model-id $(BGE_MODEL_ID) \
		--batch-size $(BGE_ENCODE_BATCH_SIZE) \
		--starting-batch-size $(BGE_ENCODE_STARTING_BATCH_SIZE) \
		--max-length $(BGE_MAX_LENGTH) \
		$(DEVICE_ARG)

.cache/indexes/%/BGE/index: .cache/embeddings/%/BGE/embeddings.jsonl
	mkdir -p .cache/indexes/$*/BGE
	python -m pyserini.index.faiss \
		--input .cache/embeddings/$*/BGE \
		--output .cache/indexes/$*/BGE \
		--dim $$(python -c 'import json; print(len(json.loads(next(open(".cache/embeddings/$*/BGE/embeddings.jsonl", encoding="utf-8")))["vector"]))')

.cache/search_runs/%/BGE.LEVEL_1.trec: .cache/indexes/%/BGE/index
	mkdir -p .cache/search_runs/$*
	for level in $(ALL_LEVELS); do \
		python -m gutenbench.search_run --encoder BGE \
			--index .cache/indexes/$*/BGE \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/BGE.$$level.trec \
			$(BGE_QUERY_CACHE_ARG) \
			--hits 100 \
			--model-id $(BGE_MODEL_ID) \
			--batch-size $(BGE_SEARCH_BATCH_SIZE) \
			--starting-batch-size $(BGE_SEARCH_STARTING_BATCH_SIZE) \
			--max-length $(BGE_MAX_LENGTH) \
			$(DEVICE_ARG); \
	done

reports/%/BGE: .cache/indexes/%/BGE/index .cache/search_runs/%/BGE.LEVEL_1.trec
	mkdir -p reports/$*/BGE
	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/BGE.$$level.trec \
			> reports/$*/BGE/$$level.txt; \
	done
