/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-RRC-Definitions"
 * 	found in "02_Aug/rrc_15_3_asn.asn1"
 * 	`asn1c -D ./Aug02 -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_SecurityModeComplete_H_
#define	_SecurityModeComplete_H_


#include <asn_application.h>

/* Including external dependencies */
#include "RRC-TransactionIdentifier.h"
#include <constr_SEQUENCE.h>
#include <constr_CHOICE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum SecurityModeComplete__criticalExtensions_PR {
	SecurityModeComplete__criticalExtensions_PR_NOTHING,	/* No components present */
	SecurityModeComplete__criticalExtensions_PR_securityModeComplete,
	SecurityModeComplete__criticalExtensions_PR_criticalExtensionsFuture
} SecurityModeComplete__criticalExtensions_PR;

/* Forward declarations */
struct SecurityModeComplete_IEs;

/* SecurityModeComplete */
typedef struct SecurityModeComplete {
	RRC_TransactionIdentifier_t	 rrc_TransactionIdentifier;
	struct SecurityModeComplete__criticalExtensions {
		SecurityModeComplete__criticalExtensions_PR present;
		union SecurityModeComplete__criticalExtensions_u {
			struct SecurityModeComplete_IEs	*securityModeComplete;
			struct SecurityModeComplete__criticalExtensions__criticalExtensionsFuture {
				
				/* Context for parsing across buffer boundaries */
				asn_struct_ctx_t _asn_ctx;
			} *criticalExtensionsFuture;
		} choice;
		
		/* Context for parsing across buffer boundaries */
		asn_struct_ctx_t _asn_ctx;
	} criticalExtensions;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} SecurityModeComplete_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_SecurityModeComplete;
extern asn_SEQUENCE_specifics_t asn_SPC_SecurityModeComplete_specs_1;
extern asn_TYPE_member_t asn_MBR_SecurityModeComplete_1[2];

#ifdef __cplusplus
}
#endif

#endif	/* _SecurityModeComplete_H_ */
#include <asn_internal.h>
