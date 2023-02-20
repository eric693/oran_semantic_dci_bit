/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "F1AP-IEs"
 * 	found in "F1.asn1"
 * 	`asn1c -D ./out -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_PacketErrorRate_H_
#define	_PacketErrorRate_H_


#include <asn_application.h>

/* Including external dependencies */
#include "PER-Scalar.h"
#include "PER-Exponent.h"
#include <constr_SEQUENCE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct ProtocolExtensionContainer;

/* PacketErrorRate */
typedef struct PacketErrorRate {
	PER_Scalar_t	 pER_Scalar;
	PER_Exponent_t	 pER_Exponent;
	struct ProtocolExtensionContainer	*iE_Extensions;	/* OPTIONAL */
	/*
	 * This type is extensible,
	 * possible extensions are below.
	 */
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} PacketErrorRate_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_PacketErrorRate;
extern asn_SEQUENCE_specifics_t asn_SPC_PacketErrorRate_specs_1;
extern asn_TYPE_member_t asn_MBR_PacketErrorRate_1[3];

#ifdef __cplusplus
}
#endif

#endif	/* _PacketErrorRate_H_ */
#include <asn_internal.h>
