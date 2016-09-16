PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean in


clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -f .DS_Store
	rm -rf dist

in: inplace

inplace:
	$(PYTHON) setup.py build_ext --inplace

doc: inplace
	$(MAKE) -C doc html

clean-doc:
	rm -rf doc/_*

cython:
	find gif -name "*.pyx" -exec $(CYTHON) {} \;

clean-hard:
	find gif -name "*.c" -exec rm {} \;
	find gif -name "*.pyc" -exec rm {} \;
	find gif -name "*.so" -exec rm {} \;
	rm -rf build

