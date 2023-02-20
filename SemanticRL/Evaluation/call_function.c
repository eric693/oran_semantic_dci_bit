#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*
int main()
{
   // Set PYTHONPATH TO working directory
   setenv("PYTHONPATH",".",1);

   PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *presult;
   Py_Initialize();// Initialize the Python Interpreter
   // pName = PyUnicode_FromString((char*)"arbName");//python3
   pName = PyUnicode_FromString((char*)"arbName");// Build the name object
   pModule = PyImport_Import(pName); // Load the module object
   pDict = PyModule_GetDict(pModule);  // pDict is a borrowed reference 
   pFunc = PyDict_GetItemString(pDict, (char*)"someFunction");// pFunc is also a borrowed reference 

   if (PyCallable_Check(pFunc))
   {
       pValue=Py_BuildValue("(z)",(char*)"something");
       PyErr_Print();
       printf("Let's give this a shot!\n");
       presult=PyObject_CallObject(pFunc,pValue);
       PyErr_Print();
   } else 
   {
       PyErr_Print();
   }
   printf("Result is %ld\n",PyLong_AsLong(presult));
   Py_DECREF(pValue);

   // Clean up
   Py_DECREF(pModule);
   Py_DECREF(pName);

   // Finish the Python Interpreter
   Py_Finalize();

    return 0;
}
*/

void showFromPython()
{
	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
	pName = PyUnicode_FromString("arbName");
	pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItemString(pDict, "someFunction");
    printf("Let's give this a shot!\n");
	PyObject_CallObject(pFunc, NULL);
}

void do_test()
{
	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
	pName = PyUnicode_FromString("arbName");
	pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItemString(pDict, "do_test");
    printf("Let's give this a shot!\n");
	PyObject_CallObject(pFunc, NULL);
}

void do_test2()
{
	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
	pName = PyUnicode_FromString("arbName");
	pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItemString(pDict, "do_test2");
    printf("Let's give this a shot!\n");
	PyObject_CallObject(pFunc, NULL);
}


void do_test3()
{
	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
	pName = PyUnicode_FromString("arbName");
	pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItemString(pDict, "do_test3");
    printf("Let's give this a shot!\n");
	PyObject_CallObject(pFunc, NULL);
}


void do_test4()
{
	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
	pName = PyUnicode_FromString("Inference_Given_Input");
	pModule = PyImport_Import(pName);
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItemString(pDict, "do_test4");
    printf("Let's give this a shot!\n");
	PyObject_CallObject(pFunc, NULL);
}



int main(void)
{	
	//調用python腳本
	Py_Initialize();
    // PySys_SetPath(L".:/usr/lib/python3.8");
	PyRun_SimpleString("import sys");
    PyRun_SimpleString("from model import LSTMEncoder, LSTMDecoder, Embeds");
	PyRun_SimpleString("sys.path.append('./')");
	showFromPython();
    do_test();
    do_test2();
    do_test3();
    //do_test4();
	Py_Finalize();
	return 0;

}
