#include <Python.h>
#include <stdio.h>

void channel()
{
	char filename[] = "channel.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/semantic/SemanticRL/Evalution/channel.py", "r");//"/home/eric/test/SemanticRL/Evalution/channel.py"
    PyRun_SimpleFile(fp, filename);
    
    
}

int main(){
    Py_Initialize();

    channel();
	Py_Finalize();
	return 0;
}