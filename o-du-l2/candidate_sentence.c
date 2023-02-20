#include <Python.h>
#include <stdio.h>

void candidate_sentence()
{
	char filename[] = "candidate_sentence.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/candidate_sentence.py", "r");
    PyRun_SimpleFile(fp, filename);
    
    
}

int main(){
    Py_Initialize();

    candidate_sentence();
	Py_Finalize();
	return 0;
}