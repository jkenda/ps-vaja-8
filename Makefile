main: src/histogram.c
	gcc -O2 -o bin/histogram src/histogram.c lib/libfreeimage.so.3 -lOpenCL

single: src/single.c
	gcc -O2 -o bin/single src/single.c lib/libfreeimage.so.3

old: src/hist_old.c
	gcc -O2 -o bin/old src/hist_old.c lib/libfreeimage.so.3 -lOpenCL