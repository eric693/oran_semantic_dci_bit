#include <Python.h>
#include <stdio.h>



void test()
{
	char filename[] = "arbName.py";
	FILE* fp;
    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);
}

int main(){
    Py_Initialize();

    test();
	Py_Finalize();
	return 0;
}
// Python initialize instance
//Py_Initialize();
//檔名要放專案底下 否則請寫完整路徑
//PyObject *obj = Py_BuildValue("s", "test.py");
//FILE *file = _Py_fopen_obj(obj, "r+");
//if(file != NULL) {
//    PyRun_SimpleFile(file, "test.py");
//}
//FILE* file = _Py_fopen("arbName.py", "r+");
//PyRun_SimpleFile(file, "arbName.py");
//Close the python instance
//Py_Finalize();