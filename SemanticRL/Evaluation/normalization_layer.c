#include <Python.h>
#include <stdio.h>

void normalization_layer()
{
	char filename[] = "normalization_layer.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/normalization_layer.py", "r");
    PyRun_SimpleFile(fp, "/home/eric/test/SemanticRL/Evaluation/normalization_layer.py");
    
    
}

int main(){
    
    Py_Initialize();

    normalization_layer();
	Py_Finalize();
	return 0;
}


