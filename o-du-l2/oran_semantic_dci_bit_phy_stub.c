#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <Python.h>

#define PORT 8089

int main(){

	int sockfd, ret;
	struct sockaddr_in serverAddr;

	int newSocket;
	struct sockaddr_in newAddr;

	socklen_t addr_size;

	char buffer[1024];
	pid_t childpid;

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0){
		printf("[-]Error in connection.\n");
		exit(1);
	}
	printf("[+]Server Socket is created.\n");

	memset(&serverAddr, '\0', sizeof(serverAddr));
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(PORT);
	serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

	ret = bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
	if(ret < 0){
		printf("[-]Error in binding.\n");
		exit(1);
	}
	printf("[+]Bind to port %d\n", 8089);

	if(listen(sockfd, 100) == 0){
		printf("[+]Listening....\n");
	}else{
		printf("[-]Error in binding.\n");
	}


	while(1){
		newSocket = accept(sockfd, (struct sockaddr*)&newAddr, &addr_size);
		if(newSocket < 0){
			exit(1);
		}
		//printf("Connection accepted from %s:%d\n", inet_ntoa(newAddr.sin_addr), ntohs(newAddr.sin_port));
		printf("Connection accepted from 127.0.0.1\n");
		printf("INFO   -->  PHY_STUB : channel_output\n");
                    Py_Initialize();
         			PyObject *noise = Py_BuildValue("s", "/home/eric/oran_semantic_dci_bit/SemanticRL/Evaluation/channel.py");
         			FILE *channel = _Py_fopen_obj(noise, "r+");
         			if(channel != NULL) {
            		PyRun_SimpleFile(channel, "/home/eric/semantic/oran_semantic_dci_bit/Evaluation/channel.py");
         			}
		printf("INFO   -->  PHY_STUB : decoder_output\n");
                    Py_Initialize();
                    PyObject *decode = Py_BuildValue("s", "/home/eric/oran_semantic_dci_bit/SemanticRL/Evaluation/decoder.py");
                    FILE *decoder = _Py_fopen_obj(decode, "r+");
                    if(decoder != NULL) {
                    PyRun_SimpleFile(decoder, "/home/eric/oran_semantic_dci_bit/SemanticRL/Evaluation/decoder.py");
                    }
		printf("INFO   -->  PHY_STUB : semanticRL_example1_candidate_sentence\n");
                    Py_Initialize();
                    PyObject *sentence = Py_BuildValue("s", "/home/eric/oran_semantic_dci_bit/SemanticRL/Evaluation/candidate_sentence.py");
                    FILE *candidate = _Py_fopen_obj(sentence, "r+");
                    if(candidate != NULL) {
                    PyRun_SimpleFile(candidate, "/home/eric/oran_semantic_dci_bit/SemanticRL/Evaluation/candidate_sentence.py\n");
                    }
		
	}
	

	close(newSocket);


	return 0;
}