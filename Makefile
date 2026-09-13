.PHONY: test audit bench clean
test:
	python -m pytest -q
audit:
	python -m cheater audit --task rule_learning --verifier final_answer_only --tau 0.1
bench:
	python -m cheater benchmark
clean:
	rm -rf runs .pytest_cache **/__pycache__
