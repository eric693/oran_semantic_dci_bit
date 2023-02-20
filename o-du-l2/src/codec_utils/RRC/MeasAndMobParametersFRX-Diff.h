/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-RRC-Definitions"
 * 	found in "02_Aug/rrc_15_3_asn.asn1"
 * 	`asn1c -D ./Aug02 -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_MeasAndMobParametersFRX_Diff_H_
#define	_MeasAndMobParametersFRX_Diff_H_


#include <asn_application.h>

/* Including external dependencies */
#include <NativeEnumerated.h>
#include <constr_SEQUENCE.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum MeasAndMobParametersFRX_Diff__ss_SINR_Meas {
	MeasAndMobParametersFRX_Diff__ss_SINR_Meas_supported	= 0
} e_MeasAndMobParametersFRX_Diff__ss_SINR_Meas;
typedef enum MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithSSB {
	MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithSSB_supported	= 0
} e_MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithSSB;
typedef enum MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithoutSSB {
	MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithoutSSB_supported	= 0
} e_MeasAndMobParametersFRX_Diff__csi_RSRP_AndRSRQ_MeasWithoutSSB;
typedef enum MeasAndMobParametersFRX_Diff__csi_SINR_Meas {
	MeasAndMobParametersFRX_Diff__csi_SINR_Meas_supported	= 0
} e_MeasAndMobParametersFRX_Diff__csi_SINR_Meas;
typedef enum MeasAndMobParametersFRX_Diff__csi_RS_RLM {
	MeasAndMobParametersFRX_Diff__csi_RS_RLM_supported	= 0
} e_MeasAndMobParametersFRX_Diff__csi_RS_RLM;
typedef enum MeasAndMobParametersFRX_Diff__ext1__handoverInterF {
	MeasAndMobParametersFRX_Diff__ext1__handoverInterF_supported	= 0
} e_MeasAndMobParametersFRX_Diff__ext1__handoverInterF;
typedef enum MeasAndMobParametersFRX_Diff__ext1__handoverLTE {
	MeasAndMobParametersFRX_Diff__ext1__handoverLTE_supported	= 0
} e_MeasAndMobParametersFRX_Diff__ext1__handoverLTE;
typedef enum MeasAndMobParametersFRX_Diff__ext1__handover_eLTE {
	MeasAndMobParametersFRX_Diff__ext1__handover_eLTE_supported	= 0
} e_MeasAndMobParametersFRX_Diff__ext1__handover_eLTE;

/* MeasAndMobParametersFRX-Diff */
typedef struct MeasAndMobParametersFRX_Diff {
	long	*ss_SINR_Meas;	/* OPTIONAL */
	long	*csi_RSRP_AndRSRQ_MeasWithSSB;	/* OPTIONAL */
	long	*csi_RSRP_AndRSRQ_MeasWithoutSSB;	/* OPTIONAL */
	long	*csi_SINR_Meas;	/* OPTIONAL */
	long	*csi_RS_RLM;	/* OPTIONAL */
	/*
	 * This type is extensible,
	 * possible extensions are below.
	 */
	struct MeasAndMobParametersFRX_Diff__ext1 {
		long	*handoverInterF;	/* OPTIONAL */
		long	*handoverLTE;	/* OPTIONAL */
		long	*handover_eLTE;	/* OPTIONAL */
		
		/* Context for parsing across buffer boundaries */
		asn_struct_ctx_t _asn_ctx;
	} *ext1;
	
	/* Context for parsing across buffer boundaries */
	asn_struct_ctx_t _asn_ctx;
} MeasAndMobParametersFRX_Diff_t;

/* Implementation */
/* extern asn_TYPE_descriptor_t asn_DEF_ss_SINR_Meas_2;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_csi_RSRP_AndRSRQ_MeasWithSSB_4;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_csi_RSRP_AndRSRQ_MeasWithoutSSB_6;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_csi_SINR_Meas_8;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_csi_RS_RLM_10;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_handoverInterF_14;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_handoverLTE_16;	// (Use -fall-defs-global to expose) */
/* extern asn_TYPE_descriptor_t asn_DEF_handover_eLTE_18;	// (Use -fall-defs-global to expose) */
extern asn_TYPE_descriptor_t asn_DEF_MeasAndMobParametersFRX_Diff;
extern asn_SEQUENCE_specifics_t asn_SPC_MeasAndMobParametersFRX_Diff_specs_1;
extern asn_TYPE_member_t asn_MBR_MeasAndMobParametersFRX_Diff_1[6];

#ifdef __cplusplus
}
#endif

#endif	/* _MeasAndMobParametersFRX_Diff_H_ */
#include <asn_internal.h>
