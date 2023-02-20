#include <Python.h>
#include <stdio.h>

void encoder()
{
	char filename[] = "encoder.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/encoder.py", "r");
    PyRun_SimpleFile(fp, filename);
    
    
}

int main(){
    Py_Initialize();

    encoder();
	Py_Finalize();

}
