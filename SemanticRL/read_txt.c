#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int main()
{

	char szTest[1000] = {0};
	int len = 0;

	FILE *input_data = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/input_data.txt", "r");
    FILE *encoder_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/encoder_output.txt", "r");
    FILE *normlize_layer_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/normlize_layer_output.txt", "r");
    FILE *channel_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/channel_output.txt", "r");
    FILE *decoder_output = fopen("/home/mwnl/o-ran_project/o-du-l2/src/5gnrmac/decoder_output.txt", "r");
    FILE *SemanticRL_example2_candidate_sentence = fopen("SemanticRL_example2_candidate_sentence.txt", "r");
    //FILE *output = fopen("/home/eric/test/SemanticRL/normlize_layer_output.txt","r");
//	if(NULL == fp)
//	{
//		printf("failed to open dos.txt\n");
//		return 1;
//	}
    printf("input_data :\n");

	while(!feof(input_data))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, input_data);
		printf("%s", szTest);
	}
    printf("\nencoder_output :\n");
    while(!feof(encoder_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, encoder_output);
		printf("%s", szTest);
	}
    printf("\nnormlize_layer_output :\n");
    while(!feof(normlize_layer_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, normlize_layer_output);
		printf("%s", szTest);
	}
    printf("\nchannel_output :\n");
    while(!feof(channel_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, channel_output);
		printf("%s", szTest);
	}
    printf("\ndecoder_output :\n");
    while(!feof(decoder_output))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, decoder_output);
		printf("%s", szTest);
	}
    printf("\nSemanticRL_example2_candidate_sentence :\n");
    while(!feof(SemanticRL_example2_candidate_sentence))
	{
		memset(szTest, 0, sizeof(szTest));
		fgets(szTest, sizeof(szTest) - 1, SemanticRL_example2_candidate_sentence);
		printf("%s", szTest);
	}
    //printf("\noutput :\n");
    //while(!feof(output))
	//{
	//	memset(szTest, 0, sizeof(szTest));
	//	fgets(szTest, sizeof(szTest) - 1, output);
	//	printf("%s", szTest);
	//}

	fclose(input_data);
    fclose(encoder_output);
    fclose(normlize_layer_output);
    fclose(channel_output);
    fclose(decoder_output);
    fclose(SemanticRL_example2_candidate_sentence);
    //fclose(output);

	printf("\n");

	return 0;
}


