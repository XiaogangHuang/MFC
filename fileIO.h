#pragma once
#pragma warning(disable:4996)
#include <algorithm>
#include <assert.h>
#include <ctime>
#include <errno.h>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


int get_dim(char* s, char* delims);
double* get_data(char* s, int dim, char* delims);
void read_data_dim_size(char* filename, int* data_dim, int* data_size, char* delims);
double* read_data(char* filename, char* delims);
double* read_data(char* filename, char* delims, int* dim, int* data_size);
