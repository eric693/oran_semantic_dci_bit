 /*******************************************************************************
 ################################################################################
 #   Copyright (c) [2017-2019] [Radisys]                                        #
 #                                                                              #
 #   Licensed under the Apache License, Version 2.0 (the "License");            #
 #   you may not use this file except in compliance with the License.           #
 #   You may obtain a copy of the License at                                    #
 #                                                                              #
 #       http://www.apache.org/licenses/LICENSE-2.0                             #
 #                                                                              #
 #   Unless required by applicable law or agreed to in writing, software        #
 #   distributed under the License is distributed on an "AS IS" BASIS,          #
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
 #   See the License for the specific language governing permissions and        #
 #   limitations under the License.                                             #
 ################################################################################
 *******************************************************************************/
#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> // bzero()
#include <sys/socket.h>
#include <math.h>
#include <unistd.h> // read(), write(), close()
#include <netinet/in.h>

#define serverPort 48763
#define serverIP "127.0.0.1"


#ifndef _LWR_MAC_FSM_H_
#define _LWR_MAC_FSM_H_

#define FAPI_UINT_8   1
#define FAPI_UINT_16  2
#define FAPI_UINT_32  4
#define INVALID_VALUE -1

#define CORESET_TYPE0 0
#define CORESET_TYPE1 1
#define CORESET_TYPE2 2
#define CORESET_TYPE3 3

#ifdef INTEL_WLS_MEM
#define WLS_MEM_FREE_PRD       10        /* Free memory after 10 slot ind */
#endif

#define FILL_FAPI_LIST_ELEM(_currElem, _nextElem, _msgType, _numMsgInBlock, _alignOffset)\
{\
   _currElem->msg_type             = (uint8_t) _msgType;\
   _currElem->num_message_in_block = _numMsgInBlock;\
   _currElem->align_offset         = (uint16_t) _alignOffset;\
   _currElem->msg_len              = _numMsgInBlock * _alignOffset;\
   _currElem->p_next               = _nextElem;\
   _currElem->p_tx_data_elm_list   = NULL;\
   _currElem->time_stamp           = 0;\
}

typedef enum{
   SI_RNTI_TYPE,
   RA_RNTI_TYPE,
   TC_RNTI_TYPE,
   C_RNTI_TYPE,
   P_RNTI_TYPE
}RntiType;

uint8_t lwr_mac_procInvalidEvt(void *msg);
uint8_t lwr_mac_procParamReqEvt(void *msg);
uint8_t lwr_mac_procParamRspEvt(void *msg);
uint8_t lwr_mac_procConfigReqEvt(void *msg);
uint8_t lwr_mac_procConfigRspEvt(void *msg);
uint8_t lwr_mac_procStartReqEvt(void *msg);
void sendToLowerMac(uint16_t, uint32_t, void *);
void procPhyMessages(uint16_t msgType, uint32_t msgSize, void *msg);
uint16_t fillDlTtiReq(SlotTimingInfo currTimingInfo);
typedef uint8_t (*lwrMacFsmHdlr)(void *);
void lwrMacLayerInit(Region region, Pool pool);
//void input_data();
//void encoder();
void channel();
void decoder();
void candidate_sentence();
#endif
void socket_client();
void socket_server();

/**********************************************************************
         End of file
**********************************************************************/

