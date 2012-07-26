default:
	rm -vf _malis.c
	rm -vf *.*o
	python setup.py build_ext --inplace
	python -c 'import _malis; print _malis.test(42)'
