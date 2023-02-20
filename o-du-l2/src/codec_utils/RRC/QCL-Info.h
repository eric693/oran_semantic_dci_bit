/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-RRC-Definitions"
 * 	found in "02_Aug/rrc_15_3_asn.asn1"
 * 	`asn1c -D ./Aug02 -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_QCL_Info_H_
#define	_QCL_Info_H_


#include <asn_application.h>

/* Including external dependencies */
#include "ServCellIndex.h"
#include "BWP-Id.h"
#include <NativeEnumerated.h>
#include "NZP-CSI-RS-ResourceId.h"
#include "SSB-Index.h"
#include <constr_CHOICE.h>
#include <constr_SEQUENCE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum QCL_Info__referenceSignal_PR {
	QCL_Info__referenceSignal_PR_NOTHING,	/* No components present */
	QCL_Info__referenceSignal_PR_csi_rs,
	QCL_Info__referenceSignal_PR_ssb
} QCL_Info__referenceSignal_PR;
typedef enum QCL_Info__qcl_Type {
	QCL_Info__qcl_Type_typeA	= 0,
	QCL_Info__qcl_Type_typeB	= 1,
	QCL_Info__qcl_Type_typeC	= 2,
	QCL_Info__qcl_Type_typeD	= 3
} e_QCL_Info__qcl_Type;

/* QCL-Info */
typedef struct QCL_Info {
	ServCellIndex_t	*cell;	/* OPTIONAL */
	BWP_Id_t	*bwp_Id;	/* OPTIONAL */
	struct QCL_Info__referenceSignal {
		QCL_Info__referenceSignal_PR present;
		union QCL_Info__referenceSignal_u {
			NZP_CSI_RS_ResourceId_t	 csi_rs;
			SSB_Index_t	 ssb;
		} choice;
		
		/* Context for parsing across buffer boundaries */
		asn_struct_ctx_t _asn_ctx;
	} referenceSignal;
	long	 qcl_Type;
	/*
	 * This type is extensible,
	 * possible extensions are below.
	 */
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} QCL_Info_t;

/* Implementation */
/* extern asn_TYPE_descriptor_t asn_DEF_qcl_Type_7;	// (Use -fall-defs-global to expose) */
extern asn_TYPE_descriptor_t asn_DEF_QCL_Info;
extern asn_SEQUENCE_specifics_t asn_SPC_QCL_Info_specs_1;
extern asn_TYPE_member_t asn_MBR_QCL_Info_1[4];

#ifdef __cplusplus
}
#endif

#endif	/* _QCL_Info_H_ */
#include <asn_internal.h>
