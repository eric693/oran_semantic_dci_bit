#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <Python.h>


void input_data();
/*
{
	char filename[] = "input_data.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/input_data.py", "r");
    PyRun_SimpleFile(fp, filename);
}
*/


void encoder();
/*
{
    char filename[] = "/home/eric/test/SemanticRL/Evaluation/encoder.py";
    FILE* fp;
    //Py_Initialize();
    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);

    //Py_Finalize();
}
*/

void channel();
/*
{
	char filename[] = "channel.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evalution/channel.py", "r");//"/home/eric/test/SemanticRL/Evalution/channel.py"
    PyRun_SimpleFile(fp, filename); 
}
*/

void decoder();
/*
{
	char filename[] = "decoder.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/decoder.py", "r");
    PyRun_SimpleFile(fp, filename);  
}
*/
void candidate_sentence();
/*
{
	char filename[] = "candidate_sentence.py";
	FILE* fp;
    fp = _Py_fopen("/home/eric/test/SemanticRL/Evaluation/candidate_sentence.py", "r");
    PyRun_SimpleFile(fp, filename); 
}
*/