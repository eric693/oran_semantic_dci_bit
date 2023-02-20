#include <Python.h>
#include <stdio.h>

void encoder()
{
	char filename[] = "encoder.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/semantic/SemanticRL/Evaluation/encoder.py", "r");
    PyRun_SimpleFile(fp, "/home/eric/semantic/SemanticRL/Evaluation/encoder.py");
    
    
}

int main(){
    
    Py_Initialize();

    encoder();
	Py_Finalize();
	return 0;
}


