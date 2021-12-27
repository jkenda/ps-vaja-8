main: src/histogram.c
	mkdir -p bin
	gcc -O2 -o bin/histogram src/histogram.c -lm -lOpenCL -Wl,-rpath,./lib -L./lib -l:"libfreeimage.so.3"

single: src/single.c
	gcc -O2 -o bin/single src/single.c lib/libfreeimage.so.3

old: src/hist_old.c
	gcc -O2 -o bin/old src/hist_old.c lib/libfreeimage.so.3 -lOpenCL

run:
	srun -n1 -G1 --reservation=fri bin/histogram