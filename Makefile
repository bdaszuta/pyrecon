.PHONY: doc docs docs-html clean

doc: docs-html

docs: docs-html

docs-html:
	rm -rf docs/build
	sphinx-build -b html docs/source docs/build/html

clean:
	rm -rf docs/build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
