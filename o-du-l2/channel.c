#include <Python.h>
#include <stdio.h>



void input_data()
{
	char filename[] = "input_data.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/input_data.py", "r");
    PyRun_SimpleFile(fp, filename);
}

void encoder()
{
	char filename[] = "encoder.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/encoder.py", "r");
    PyRun_SimpleFile(fp, filename);   
}

void channel()
{
	char filename[] = "channel.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evalution/channel.py", "r");
    PyRun_SimpleFile(fp, filename); 
}

void decoder()
{
	char filename[] = "decoder.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/decoder.py", "r");
    PyRun_SimpleFile(fp, filename);  
}

void candidate_sentence()
{
	char filename[] = "candidate_sentence.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/candidate_sentence.py", "r");
    PyRun_SimpleFile(fp, filename); 
}

int main(){
    int i;
    Py_Initialize();
    input_data();
    encoder();
    //channel();
    //decoder();
    //candidate_sentence();
    
	Py_Finalize();
	
}