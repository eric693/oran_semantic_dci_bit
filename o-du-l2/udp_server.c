#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define serverPort 48763

/*
char *convert(char *src) {
    char *iter = src;
    char *result = malloc(sizeof(src));
    char *it = result;
    if (iter == NULL) return iter;

    while (*iter) {
        *it++ = *iter++ & ~0x20;
    }
    return result;
}

int main(int argc , char *argv[])
{
    // message buffer
    char buf[1024] = {0};

    // 建立 socket
    int socket_fd = socket(PF_INET , SOCK_DGRAM , 0);
    if (socket_fd < 0){
        printf("Fail to create a socket.");
    }
    
    // server 地址
    struct sockaddr_in serverAddr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(serverPort)
    };
    
    // 將建立的 socket 綁定到 serverAddr 指定的 port
    if (bind(socket_fd, (const struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Bind socket failed!");
        close(socket_fd);
        exit(0);
    }
    
    printf("Server ready!\n");

    struct sockaddr_in clientAddr;
    int len = sizeof(clientAddr);
    while (1) {
        // 當有人使用 UDP 協定送資料到 48763 port
        // 會觸發 recvfrom()，並且把來源資料寫入 clientAddr 當中
        if (recvfrom(socket_fd, buf, sizeof(buf), 0, (struct sockaddr *)&clientAddr, &len) < 0) {
            break;
        }

        // 收到 exit 指令就關閉 server
        if (strcmp(buf, "exit") == 0) {
            printf("get exit order, closing the server...\n");
            break;
        }
        
        // 將收到的英文字母換成大寫
        char *conv = convert(buf);

        // 顯示資料來源，原本資料，以及修改後的資料
        printf("get message from [%s:%d]: ", inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));
        printf("%s -> %s\n", buf, conv);

        // 根據 clientAddr 的資訊，回傳至 client 端                
        sendto(socket_fd, conv, sizeof(conv), 0, (struct sockaddr *)&clientAddr, sizeof(clientAddr));

        // 清空 message buffer
        memset(buf, 0, sizeof(buf));
        free(conv);
    }
    // 關閉 socket，並檢查是否關閉成功
    if (close(socket_fd) < 0) {
        perror("close socket failed!");
    }
    
    return 0;
}

*/

// Server side implementation of UDP client-server model
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
	char *hello = "Hello from server";
	struct sockaddr_in servaddr, cliaddr;
		
	// Creating socket file descriptor
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
		perror("socket creation failed");
		exit(EXIT_FAILURE);
	}
		
	memset(&servaddr, 0, sizeof(servaddr));
	memset(&cliaddr, 0, sizeof(cliaddr));
		
	// Filling server information
	servaddr.sin_family = AF_INET; // IPv4
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(PORT);
		
	// Bind the socket with the server address
	if ( bind(sockfd, (const struct sockaddr *)&servaddr,
			sizeof(servaddr)) < 0 )
	{
		perror("bind failed");
		exit(EXIT_FAILURE);
	}
		
	int len, n;
	
	len = sizeof(cliaddr); //len is value/result
	
	n = recvfrom(sockfd, (char *)buffer, MAXLINE,
				MSG_WAITALL, ( struct sockaddr *) &cliaddr,
				&len);
	buffer[n] = '\0';
	printf("Client : %s\n", buffer);
	sendto(sockfd, (const char *)hello, strlen(hello),
		MSG_CONFIRM, (const struct sockaddr *) &cliaddr,
			len);
	printf("Hello message sent.\n");

    
    /*
    char szTest[1000] = {0};

    FILE *channel_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_channel_output.txt", "r");
    FILE *decoder_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_decoder_output.txt", "r");
    FILE *semanticRL_example2_candidate_sentence = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/semanticRL_example2_candidate_sentence.txt", "r");
      
    while(!feof(channel_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, channel_output);
		printf("%s", szTest);
	}
    
    while(!feof(decoder_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, decoder_output);
		printf("%s", szTest);
    }
        
    while(!feof(semanticRL_example2_candidate_sentence))
	{
	    memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, semanticRL_example2_candidate_sentence);
		printf("%s", szTest);
	}
    fclose(channel_output);
    fclose(decoder_output);
    fclose(semanticRL_example2_candidate_sentence);
	
    */
	return 0;
}
