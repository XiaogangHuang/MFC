# MFC

This is the C++ source code for MFC and the corresponding executable.

Descriptions:

1) "Usage: ./MFC [-a alg] [-nn k] [-c clusters] [-df data] [-pre preference] [-r normalization] [-t tree]"

Example:

./MFC -nn 12 -c 3 -df iris -r 1


2) The input dataset needs to be preprocessed into a text file with the following format:

* In each line: its coordinates and the point's id, where the id is an integer in the range [0,  n-1].

* In total, there are n lines where the numbers at each line are separated by a space. 

For example, a 2-dimensional data set containing 4 points:
 
9.3498     56.7408     17.0527     0

9.3501     56.7406     17.6148     1

9.3505     56.7405     18.0835     2

9.3508     56.7404     18.2794     3

3) The output file contains cluster labels 
