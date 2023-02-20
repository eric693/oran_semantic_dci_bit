#include <Python.h>
#include <stdio.h>

void input_data()
{
	char filename[] = "input_data.py";
	FILE* fp;
    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    
}

int main(){
    Py_Initialize();
    
    input_data();
	Py_Finalize();
    return 0;
}
