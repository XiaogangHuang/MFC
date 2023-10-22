#include "ClusterAnalysis.h"

/*
 *  Get next word from a string.
 */
void getNextWord(char* str, char* word) {
	// Jump over all blanks
	while (*str == ' ') {
		str++;
	}

	while (*str != ' ' && *str != '\0') {
		*word = *str;
		str++;
		word++;
	}

	*word = '\0';
}

int main(int argc, char** argv)
{
	int k = 0, nc = 2, alg = 0, clu_num = -1, r = 0, o_tree = 0;
	double pre = 0.9;
	string rootfile = "data\\";
	string suffixes = ".txt";
	string myfile = "";
	char str[20] = { '\0' };
	cout << "Usage:\n"
		<< "  .\\ClusterTree_KD [-a alg] [-nn k] [-c clusters] [-df data] [-pre preference] [-r normalization] [-t tree]\n"
		<< "  where:\n"
		<< "    alg				the algorithm you choose (default = 0)\n"
		<< "    k				number of nearest neighbors per query\n"
		<< "    clusters			number of clusters\n"
		<< "    data			name of file containing data points\n"
		<< "    preference			preference to macro consolidation factor (default = 0.9)\n"
		<< "    normalization		minmax normalization\n"
		<< "    tree			output hierarchical structure (default = 0)\n"
		<< "Results are sent to the standard output.\n\n"
		<< "0. Run ClusterTree.\n"
		<< "  .\\ClusterTree_KD -a 0 -nn 12 -c 3 -df iris\n\n"
		<< "1. Run ClusterTree_MNN.\n"
		<< "  .\\ClusterTree_KD -a 1 -nn 12 -c 3 -df iris\n\n"
		<< "2. Run LDP-MST\n"
		<< "  .\\ClusterTree_KD -a 2 -c 3 -df iris\n\n";
	if (argc > 0)
	{
		int i = 1;
		while (i < argc) {							// read arguments
			if (!strcmp(argv[i], "-a")) {		// -a option
				alg = atoi(argv[++i]);				// choose algorithm
			}
			else if (!strcmp(argv[i], "-pre")) {		// -r option
				sscanf_s(argv[++i], "%lf", &pre);		// get radius
			}
			else if (!strcmp(argv[i], "-nn")) {		// -nn option
				k = atoi(argv[++i]);				// get number of near neighbors
			}
			else if (!strcmp(argv[i], "-t")) {		// -nn option
				o_tree = atoi(argv[++i]);				// get number of near neighbors
			}
			else if (!strcmp(argv[i], "-c")) {		// -nn option
				clu_num = atoi(argv[++i]);			// get number of clusters
			}
			else if (!strcmp(argv[i], "-nc")) {		// -nc option
				nc = atoi(argv[++i]);				// get number of clusters
			}
			else if (!strcmp(argv[i], "-df")) {		// -df option
				getNextWord(argv[++i], str);
			}
			else if (!strcmp(argv[i], "-r")) {		// -nn option
				r = atoi(argv[++i]);				// get number of near neighbors
			}
			else {									// illegal syntax
				cerr << "Unrecognized option.\n";
				exit(1);
			}
			i++;
		}
		if (str[0] == '\0')
		{
			cerr << "name of file error.\n";
			exit(1);
		}
		myfile = str;
	}
	string filename = rootfile + myfile + suffixes;
	cout << filename << " " << k << " " << pre << endl;
	ClusterAnalysis myClusterAnalysis;                      //Clustering algorithm object declaration
	if (alg != 2)
	{
		myClusterAnalysis.Init((char*)filename.c_str(), k, alg, clu_num, r, o_tree, pre);
		printf("Clusting the data...\n");
		CYW_TIMER build_timer;
		build_timer.start_my_timer();
		myClusterAnalysis.Running();                   //Perform ClusterTree
		build_timer.stop_my_timer();
		printf("Running time = %.4f\n", build_timer.get_my_timer());
	}
	else
	{
		myClusterAnalysis.Init((char*)filename.c_str(), clu_num);	//Initialization
		printf("Clusting the data...\n");
		CYW_TIMER build_timer;
		build_timer.start_my_timer();
		myClusterAnalysis.Running_LDP_MST();                   //Perform ClusterTree
		build_timer.stop_my_timer();
		printf("Running time = %.4f\n", build_timer.get_my_timer());
	}
	myClusterAnalysis.WriteToFile();                         //Save the result
	return 0;
}
