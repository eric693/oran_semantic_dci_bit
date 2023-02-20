/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "F1AP-IEs"
 * 	found in "F1.asn1"
 * 	`asn1c -D ./out -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#include "EUTRA-Coex-Mode-Info.h"

#include "EUTRA-Coex-FDD-Info.h"
#include "EUTRA-Coex-TDD-Info.h"
static asn_oer_constraints_t asn_OER_type_EUTRA_Coex_Mode_Info_constr_1 CC_NOTUSED = {
	{ 0, 0 },
	-1};
asn_per_constraints_t asn_PER_type_EUTRA_Coex_Mode_Info_constr_1 CC_NOTUSED = {
	{ APC_CONSTRAINED | APC_EXTENSIBLE,  1,  1,  0,  1 }	/* (0..1,...) */,
	{ APC_UNCONSTRAINED,	-1, -1,  0,  0 },
	0, 0	/* No PER value map */
};
asn_TYPE_member_t asn_MBR_EUTRA_Coex_Mode_Info_1[] = {
	{ ATF_POINTER, 0, offsetof(struct EUTRA_Coex_Mode_Info, choice.fDD),
		(ASN_TAG_CLASS_CONTEXT | (0 << 2)),
		-1,	/* IMPLICIT tag at current level */
		&asn_DEF_EUTRA_Coex_FDD_Info,
		0,
		{ 0, 0, 0 },
		0, 0, /* No default value */
		"fDD"
		},
	{ ATF_POINTER, 0, offsetof(struct EUTRA_Coex_Mode_Info, choice.tDD),
		(ASN_TAG_CLASS_CONTEXT | (1 << 2)),
		-1,	/* IMPLICIT tag at current level */
		&asn_DEF_EUTRA_Coex_TDD_Info,
		0,
		{ 0, 0, 0 },
		0, 0, /* No default value */
		"tDD"
		},
};
static const asn_TYPE_tag2member_t asn_MAP_EUTRA_Coex_Mode_Info_tag2el_1[] = {
    { (ASN_TAG_CLASS_CONTEXT | (0 << 2)), 0, 0, 0 }, /* fDD */
    { (ASN_TAG_CLASS_CONTEXT | (1 << 2)), 1, 0, 0 } /* tDD */
};
asn_CHOICE_specifics_t asn_SPC_EUTRA_Coex_Mode_Info_specs_1 = {
	sizeof(struct EUTRA_Coex_Mode_Info),
	offsetof(struct EUTRA_Coex_Mode_Info, _asn_ctx),
	offsetof(struct EUTRA_Coex_Mode_Info, present),
	sizeof(((struct EUTRA_Coex_Mode_Info *)0)->present),
	asn_MAP_EUTRA_Coex_Mode_Info_tag2el_1,
	2,	/* Count of tags in the map */
	0, 0,
	2	/* Extensions start */
};
asn_TYPE_descriptor_t asn_DEF_EUTRA_Coex_Mode_Info = {
	"EUTRA-Coex-Mode-Info",
	"EUTRA-Coex-Mode-Info",
	&asn_OP_CHOICE,
	0,	/* No effective tags (pointer) */
	0,	/* No effective tags (count) */
	0,	/* No tags (pointer) */
	0,	/* No tags (count) */
	{ &asn_OER_type_EUTRA_Coex_Mode_Info_constr_1, &asn_PER_type_EUTRA_Coex_Mode_Info_constr_1, CHOICE_constraint },
	asn_MBR_EUTRA_Coex_Mode_Info_1,
	2,	/* Elements count */
	&asn_SPC_EUTRA_Coex_Mode_Info_specs_1	/* Additional specs */
};

