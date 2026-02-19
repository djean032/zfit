SHELL := cmd.exe
.SHELLFLAGS := /c

all:
	if not exist obj mkdir obj
	if not exist mod mkdir mod
	gfortran -O3 -std=legacy -fallow-argument-mismatch -shared -Jmod -o libzfit.dll src/minpack.f90 src/global.f90 src/laser.f90 src/chem.f90 src/zfit.f90 src/opkdmain.f src/opkda1.f src/opkda2.f -static-libgcc -static-libgfortran -static-libquadmath

clean:
	if exist obj rmdir /s /q obj
	if exist mod rmdir /s /q mod
	del /q libzfit.a libzfit.dll *.mod

.PHONY: all clean
