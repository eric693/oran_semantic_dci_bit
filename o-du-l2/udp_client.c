#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define serverPort 48763
#define serverIP "127.0.0.1"
/*
int main() 
{
    // message buffer
    char buf[1024] = {0};
    char recvbuf[1024] = {0};
    
    // 建立 socket
    int socket_fd = socket(PF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) {
        printf("Create socket fail!\n");
        return -1;
    }

    // server 地址
    struct sockaddr_in serverAddr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = inet_addr(serverIP),
        .sin_port = htons(serverPort)
    };
    int len = sizeof(serverAddr);

    while (1) {
        // 輸入資料到 buffer
        printf("Please input your message: ");
        scanf("%s", buf);

        // 傳送到 server 端
        sendto(socket_fd, buf, sizeof(buf), 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
        
        // 接收到 exit 指令就退出迴圈
        if (strcmp(buf, "exit") == 0) 
            break;

        // 清空 message buffer
        memset(buf, 0, sizeof(buf));

        // 等待 server 回傳轉成大寫的資料
        if (recvfrom(socket_fd, recvbuf, sizeof(recvbuf), 0, (struct sockaddr *)&serverAddr, &len) < 0) {
            printf("recvfrom data from %s:%d, failed!\n", inet_ntoa(serverAddr.sin_addr), ntohs(serverAddr.sin_port));
            break;
        }
        
        // 顯示 server 地址，以及收到的資料
        printf("get receive message from [%s:%d]: %s\n", inet_ntoa(serverAddr.sin_addr), ntohs(serverAddr.sin_port), recvbuf);
        memset(recvbuf, 0, sizeof(recvbuf));
    }
    // 關閉 socket，並檢查是否關閉成功
    if (close(socket_fd) < 0) {
        perror("close socket failed!");
    }

    return 0;
}

*/

// Client side implementation of UDP client-server model
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
	
#define PORT	 8080
#define MAXLINE 1024
	
// Driver code
int main() {
	int sockfd;
	char buffer[MAXLINE];
	char *hello = "Hello from client";
	struct sockaddr_in	 servaddr;
	
	// Creating socket file descriptor
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
		perror("socket creation failed");
		exit(EXIT_FAILURE);
	}
	
	memset(&servaddr, 0, sizeof(servaddr));
		
	// Filling server information
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT);
	servaddr.sin_addr.s_addr = INADDR_ANY;
		
	int n, len;
		
	sendto(sockfd, (const char *)hello, strlen(hello),
		MSG_CONFIRM, (const struct sockaddr *) &servaddr,
			sizeof(servaddr));
	printf("Hello message sent.\n");
			
	n = recvfrom(sockfd, (char *)buffer, MAXLINE,
				MSG_WAITALL, (struct sockaddr *) &servaddr,
				&len);
	buffer[n] = '\0';
	printf("Server : %s\n", buffer);
	
    char szTest[1000] = {0};

    FILE *input_data = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_input_data.txt", "r");
    FILE *encoder_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_encoder_output.txt", "r");
    FILE *normlize_layer_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_normlize_layer_output.txt", "r");
     

	while(!feof(input_data))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, input_data);
		printf("%s", szTest);
	}
      
    while(!feof(encoder_output))
	{
	    memset(szTest, 0, sizeof(szTest));
	   	fgets(szTest, sizeof(szTest) - 1, encoder_output);
	   	printf("%s", szTest);
	}
      
    while(!feof(normlize_layer_output))
   	{
   		memset(szTest, 0, sizeof(szTest));
   		fgets(szTest, sizeof(szTest) - 1, normlize_layer_output);
   		printf("%s", szTest);
   	}
    fclose(input_data);
    fclose(encoder_output);
    fclose(normlize_layer_output);

	//close(sockfd);


	return 0;
}
