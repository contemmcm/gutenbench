BM25/%: reports/%/BM25
	cat reports/$*/BM25/LEVEL_1.txt
	cat reports/$*/BM25/LEVEL_2.txt
	cat reports/$*/BM25/LEVEL_3.txt
	cat reports/$*/BM25/LEVEL_4.txt

.PRECIOUS: reports/%/BM25 .cache/indexes/%/BM25 .cache/search_runs/%/BM25.LEVEL_1.trec

.cache/indexes/%/BM25: corpora/%
	mkdir -p .cache/indexes/$*/BM25
	python -m pyserini.index.lucene \
		--collection JsonCollection \
		--input corpora/$* \
		--index .cache/indexes/$*/BM25 \
		--generator DefaultLuceneDocumentGenerator \
		--threads 1 \
		--storePositions --storeDocvectors --storeRaw

.cache/search_runs/%/BM25.LEVEL_1.trec: .cache/indexes/%/BM25

	mkdir -p .cache/search_runs/$*

	for level in $(ALL_LEVELS); do \
		python -m pyserini.search.lucene \
			--index .cache/indexes/$*/BM25 \
			--topics dataset/queries_$$level.tsv \
			--output .cache/search_runs/$*/BM25.$$level.trec \
			--output-format trec \
			--hits 100 \
			--bm25; \
	done

reports/%/BM25: .cache/search_runs/%/BM25.LEVEL_1.trec
	mkdir -p reports/$*/BM25

	for level in $(LEVELS); do \
		python -m pyserini.eval.trec_eval -c -m map -m ndcg_cut.10 -m recall.100 \
			dataset/qrels_$$level.txt \
			.cache/search_runs/$*/BM25.$$level.trec \
			> reports/$*/BM25/$$level.txt; \
	done
