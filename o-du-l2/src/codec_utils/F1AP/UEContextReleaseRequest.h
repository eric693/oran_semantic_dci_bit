/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "F1AP-PDU-Contents"
 * 	found in "F1.asn1"
 * 	`asn1c -D ./out -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_UEContextReleaseRequest_H_
#define	_UEContextReleaseRequest_H_


#include <asn_application.h>

/* Including external dependencies */
#include "ProtocolIE-Container.h"
#include <constr_SEQUENCE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* UEContextReleaseRequest */
typedef struct UEContextReleaseRequest {
	ProtocolIE_Container_4587P17_t	 protocolIEs;
	/*
	 * This type is extensible,
	 * possible extensions are below.
	 */
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} UEContextReleaseRequest_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_UEContextReleaseRequest;
extern asn_SEQUENCE_specifics_t asn_SPC_UEContextReleaseRequest_specs_1;
extern asn_TYPE_member_t asn_MBR_UEContextReleaseRequest_1[1];

#ifdef __cplusplus
}
#endif

#endif	/* _UEContextReleaseRequest_H_ */
#include <asn_internal.h>
