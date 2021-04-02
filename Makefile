requirements:
	pip-compile requirements.in > requirements.txt
	pip-sync requirements.txt

dev-requirements:
	pip-compile dev-requirements.in > dev-requirements.txt
	pip-sync dev-requirements.txt

test:
	python -m table_extractor.run run-sequentially test_resources/ocr_understanding_hipaa_rus.pdf .


prepare:
	autoflake table_extractor --in-place --remove-all-unused-imports --remove-unused-variables -r
	black -S table_extractor
	isort --multi-line=3 --trailing-comma table_extractor