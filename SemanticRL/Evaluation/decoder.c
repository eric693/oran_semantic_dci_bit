#include <Python.h>
#include <stdio.h>

void decoder()
{
	char filename[] = "decoder.py";
	FILE* fp;
    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);
    
    
}

int main(){
    Py_Initialize();

    decoder();
	Py_Finalize();
	return 0;
}