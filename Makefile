requirements:
	pip-compile requirements.in > requirements.txt
	pip-sync requirements.txt

test:
	python -m table_extractor.run run-sequentially test_resources/ocr_understanding_hipaa_rus.pdf .