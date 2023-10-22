#include "fileIO.h"

int get_dim(char* s, char* delims){
    char *val_str = NULL;
    val_str = strtok( s, delims );
    int dim=0;
    while( val_str != NULL ) {
        dim++;
        val_str = strtok( NULL, delims );
    }
    return dim;
}

double* get_data(char* s, int dim,char* delims){
    double* temp= (double*) malloc (dim*sizeof(double));
    char *val_str = NULL;
    val_str = strtok( s, delims );
    int counter=0;
    while( val_str != NULL ) {
        temp[counter]=atof(val_str);
        counter++;
        val_str = strtok( NULL, delims );
    }
    return temp;
}

void read_data_dim_size(char* filename, int* data_dim, int* data_size, char* delims){
    int n_size=0;
    int dim=0;
    char s[10000];
    freopen(filename,"r",stdin);
    while(gets_s(s))
    {
        if (dim==0)
           dim=get_dim(s,delims);
        n_size ++;
    }
    *data_dim=dim;
    *data_size=n_size;
    fclose(stdin);
}

double* read_data(char* filename, char* delims){
    int dim, n_size;
    read_data_dim_size(filename,&dim, &n_size, delims);

    double* data= (double*) malloc (n_size*dim*sizeof(double));
    freopen(filename,"r",stdin);
    int counter=0;
    char s[10000];
    while(gets_s(s))
    {
        double* tmp_data= get_data( s, dim,delims);
        memcpy(data+counter*dim,tmp_data,dim*sizeof(double));
        counter++;
        free(tmp_data);
    }
    fclose(stdin);

    return data;
}

double* read_data(char* filename, char* delims, int* dim, int* data_size){
    read_data_dim_size(filename,dim, data_size, delims);

    double* data= (double*) malloc ((*data_size)*(*dim)*sizeof(double));
    freopen(filename,"r",stdin);
    int counter=0;
    char s[10000];
    while(gets_s(s))
    {
        double* tmp_data= get_data( s,*dim,delims);
        memcpy(data+counter*(*dim),tmp_data,(*dim)*sizeof(double));
        counter++;
        free(tmp_data);
    }
    fclose(stdin);

    return data;
}
