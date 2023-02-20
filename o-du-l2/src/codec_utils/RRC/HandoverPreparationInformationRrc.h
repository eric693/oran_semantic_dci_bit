/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-InterNodeDefinitions"
 * 	found in "02_Aug/rrc_15_3_asn.asn1"
 * 	`asn1c -D ./Aug02 -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_HandoverPreparationInformationRrc_H_
#define	_HandoverPreparationInformationRrc_H_


#include <asn_application.h>

/* Including external dependencies */
#include <NULL.h>
#include <constr_CHOICE.h>
#include <constr_SEQUENCE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum HandoverPreparationInformationRrc__criticalExtensions_PR {
	HandoverPreparationInformationRrc__criticalExtensions_PR_NOTHING,	/* No components present */
	HandoverPreparationInformationRrc__criticalExtensions_PR_c1,
	HandoverPreparationInformationRrc__criticalExtensions_PR_criticalExtensionsFuture
} HandoverPreparationInformationRrc__criticalExtensions_PR;
typedef enum HandoverPreparationInformationRrc__criticalExtensions__c1_PR {
	HandoverPreparationInformationRrc__criticalExtensions__c1_PR_NOTHING,	/* No components present */
	HandoverPreparationInformationRrc__criticalExtensions__c1_PR_handoverPreparationInformation,
	HandoverPreparationInformationRrc__criticalExtensions__c1_PR_spare3,
	HandoverPreparationInformationRrc__criticalExtensions__c1_PR_spare2,
	HandoverPreparationInformationRrc__criticalExtensions__c1_PR_spare1
} HandoverPreparationInformationRrc__criticalExtensions__c1_PR;

/* Forward declarations */
struct HandoverPreparationInformationRrc_IEs;

/* HandoverPreparationInformationRrc */
typedef struct HandoverPreparationInformationRrc {
	struct HandoverPreparationInformationRrc__criticalExtensions {
		HandoverPreparationInformationRrc__criticalExtensions_PR present;
		union HandoverPreparationInformationRrc__criticalExtensions_u {
			struct HandoverPreparationInformationRrc__criticalExtensions__c1 {
				HandoverPreparationInformationRrc__criticalExtensions__c1_PR present;
				union HandoverPreparationInformationRrc__criticalExtensions__c1_u {
					struct HandoverPreparationInformationRrc_IEs	*handoverPreparationInformation;
					NULL_t	 spare3;
					NULL_t	 spare2;
					NULL_t	 spare1;
				} choice;
				
				/* Context for parsing across buffer boundaries */
				asn_struct_ctx_t _asn_ctx;
			} *c1;
			struct HandoverPreparationInformationRrc__criticalExtensions__criticalExtensionsFuture {
				
				/* Context for parsing across buffer boundaries */
				asn_struct_ctx_t _asn_ctx;
			} *criticalExtensionsFuture;
		} choice;
		
		/* Context for parsing across buffer boundaries */
		asn_struct_ctx_t _asn_ctx;
	} criticalExtensions;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} HandoverPreparationInformationRrc_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_HandoverPreparationInformationRrc;

#ifdef __cplusplus
}
#endif

#endif	/* _HandoverPreparationInformationRrc_H_ */
#include <asn_internal.h>