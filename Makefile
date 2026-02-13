SHELL := cmd.exe
.SHELLFLAGS := /c

all:
	if not exist obj mkdir obj
	if not exist mod mkdir mod
	gfortran -O3 -c -Jmod -o obj/minpack.f90.o src/minpack.f90
	gfortran -O3 -c -Jmod -o obj/global.f90.o src/global.f90
	gfortran -O3 -c -Jmod -o obj/laser.f90.o src/laser.f90
	gfortran -O3 -c -Jmod -o obj/chem.f90.o src/chem.f90
	gfortran -O3 -c -Jmod -o obj/zfit.f90.o src/zfit.f90
	gfortran -O3 -c -std=legacy -fallow-argument-mismatch -Jmod -o obj/opkdmain.f.o src/opkdmain.f
	gfortran -O3 -c -std=legacy -fallow-argument-mismatch -Jmod -o obj/opkda1.f.o src/opkda1.f
	gfortran -O3 -c -std=legacy -fallow-argument-mismatch -Jmod -o obj/opkda2.f.o src/opkda2.f
	ar rcs libzfit.a obj/minpack.f90.o obj/global.f90.o obj/laser.f90.o obj/chem.f90.o obj/zfit.f90.o obj/opkdmain.f.o obj/opkda1.f.o obj/opkda2.f.o
	gfortran -shared -o libzfit.dll obj/minpack.f90.o obj/global.f90.o obj/laser.f90.o obj/chem.f90.o obj/zfit.f90.o obj/opkdmain.f.o obj/opkda1.f.o obj/opkda2.f.o -static-libgcc -static-libgfortran -static-libquadmath

clean:
	if exist obj rmdir /s /q obj
	if exist mod rmdir /s /q mod
	del /q libzfit.a libzfit.dll *.mod

.PHONY: all clean
