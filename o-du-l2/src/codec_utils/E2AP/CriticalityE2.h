/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "E2AP-CommonDataTypes"
 * 	found in "2022_E2AP.asn1"
 * 	`asn1c -D ./E2AP/ -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_CriticalityE2_H_
#define	_CriticalityE2_H_


#include <asn_application.h>

/* Including external dependencies */
#include <NativeEnumerated.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum CriticalityE2 {
	CriticalityE2_reject	= 0,
	CriticalityE2_ignore	= 1,
	CriticalityE2_notify	= 2
} e_CriticalityE2;

/* CriticalityE2 */
typedef long	 CriticalityE2_t;

/* Implementation */
extern asn_per_constraints_t asn_PER_type_CriticalityE2_constr_1;
extern asn_TYPE_descriptor_t asn_DEF_CriticalityE2;
extern const asn_INTEGER_specifics_t asn_SPC_CriticalityE2_specs_1;
asn_struct_free_f CriticalityE2_free;
asn_struct_print_f CriticalityE2_print;
asn_constr_check_f CriticalityE2_constraint;
ber_type_decoder_f CriticalityE2_decode_ber;
der_type_encoder_f CriticalityE2_encode_der;
xer_type_decoder_f CriticalityE2_decode_xer;
xer_type_encoder_f CriticalityE2_encode_xer;
oer_type_decoder_f CriticalityE2_decode_oer;
oer_type_encoder_f CriticalityE2_encode_oer;
per_type_decoder_f CriticalityE2_decode_uper;
per_type_encoder_f CriticalityE2_encode_uper;
per_type_decoder_f CriticalityE2_decode_aper;
per_type_encoder_f CriticalityE2_encode_aper;

#ifdef __cplusplus
}
#endif

#endif	/* _CriticalityE2_H_ */
#include <asn_internal.h>
