/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "F1AP-IEs"
 * 	found in "F1.asn1"
 * 	`asn1c -D ./out -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#ifndef	_TransmissionActionIndicator_H_
#define	_TransmissionActionIndicator_H_


#include <asn_application.h>

/* Including external dependencies */
#include <NativeEnumerated.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependencies */
typedef enum TransmissionActionIndicator {
	TransmissionActionIndicator_stop	= 0,
	/*
	 * Enumeration is extensible
	 */
	TransmissionActionIndicator_restart	= 1
} e_TransmissionActionIndicator;

/* TransmissionActionIndicator */
typedef long	 TransmissionActionIndicator_t;

/* Implementation */
extern asn_TYPE_descriptor_t asn_DEF_TransmissionActionIndicator;
asn_struct_free_f TransmissionActionIndicator_free;
asn_struct_print_f TransmissionActionIndicator_print;
asn_constr_check_f TransmissionActionIndicator_constraint;
ber_type_decoder_f TransmissionActionIndicator_decode_ber;
der_type_encoder_f TransmissionActionIndicator_encode_der;
xer_type_decoder_f TransmissionActionIndicator_decode_xer;
xer_type_encoder_f TransmissionActionIndicator_encode_xer;
oer_type_decoder_f TransmissionActionIndicator_decode_oer;
oer_type_encoder_f TransmissionActionIndicator_encode_oer;
per_type_decoder_f TransmissionActionIndicator_decode_uper;
per_type_encoder_f TransmissionActionIndicator_encode_uper;
per_type_decoder_f TransmissionActionIndicator_decode_aper;
per_type_encoder_f TransmissionActionIndicator_encode_aper;

#ifdef __cplusplus
}
#endif

#endif	/* _TransmissionActionIndicator_H_ */
#include <asn_internal.h>
