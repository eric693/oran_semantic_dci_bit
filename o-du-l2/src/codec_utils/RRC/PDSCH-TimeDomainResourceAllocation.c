/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-RRC-Definitions"
 * 	found in "02_Aug/rrc_15_3_asn.asn1"
 * 	`asn1c -D ./Aug02 -fcompound-names -fno-include-deps -findirect-choice -gen-PER -no-gen-example`
 */

#include "PDSCH-TimeDomainResourceAllocation.h"

/*
 * This type is implemented using NativeEnumerated,
 * so here we adjust the DEF accordingly.
 */
static int
memb_k0_constraint_1(const asn_TYPE_descriptor_t *td, const void *sptr,
			asn_app_constraint_failed_f *ctfailcb, void *app_key) {
	long value;
	
	if(!sptr) {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: value not given (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
	
	value = *(const long *)sptr;
	
	if((value >= 0 && value <= 32)) {
		/* Constraint check succeeded */
		return 0;
	} else {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: constraint failed (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
}

static int
memb_startSymbolAndLength_constraint_1(const asn_TYPE_descriptor_t *td, const void *sptr,
			asn_app_constraint_failed_f *ctfailcb, void *app_key) {
	long value;
	
	if(!sptr) {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: value not given (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
	
	value = *(const long *)sptr;
	
	if((value >= 0 && value <= 127)) {
		/* Constraint check succeeded */
		return 0;
	} else {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: constraint failed (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
}

static asn_oer_constraints_t asn_OER_type_mappingType_constr_3 CC_NOTUSED = {
	{ 0, 0 },
	-1};
static asn_per_constraints_t asn_PER_type_mappingType_constr_3 CC_NOTUSED = {
	{ APC_CONSTRAINED,	 1,  1,  0,  1 }	/* (0..1) */,
	{ APC_UNCONSTRAINED,	-1, -1,  0,  0 },
	0, 0	/* No PER value map */
};
static asn_oer_constraints_t asn_OER_memb_k0_constr_2 CC_NOTUSED = {
	{ 1, 1 }	/* (0..32) */,
	-1};
static asn_per_constraints_t asn_PER_memb_k0_constr_2 CC_NOTUSED = {
	{ APC_CONSTRAINED,	 6,  6,  0,  32 }	/* (0..32) */,
	{ APC_UNCONSTRAINED,	-1, -1,  0,  0 },
	0, 0	/* No PER value map */
};
static asn_oer_constraints_t asn_OER_memb_startSymbolAndLength_constr_6 CC_NOTUSED = {
	{ 1, 1 }	/* (0..127) */,
	-1};
static asn_per_constraints_t asn_PER_memb_startSymbolAndLength_constr_6 CC_NOTUSED = {
	{ APC_CONSTRAINED,	 7,  7,  0,  127 }	/* (0..127) */,
	{ APC_UNCONSTRAINED,	-1, -1,  0,  0 },
	0, 0	/* No PER value map */
};
static const asn_INTEGER_enum_map_t asn_MAP_mappingType_value2enum_3[] = {
	{ 0,	5,	"typeA" },
	{ 1,	5,	"typeB" }
};
static const unsigned int asn_MAP_mappingType_enum2value_3[] = {
	0,	/* typeA(0) */
	1	/* typeB(1) */
};
static const asn_INTEGER_specifics_t asn_SPC_mappingType_specs_3 = {
	asn_MAP_mappingType_value2enum_3,	/* "tag" => N; sorted by tag */
	asn_MAP_mappingType_enum2value_3,	/* N => "tag"; sorted by N */
	2,	/* Number of elements in the maps */
	0,	/* Enumeration is not extensible */
	1,	/* Strict enumeration */
	0,	/* Native long size */
	0
};
static const ber_tlv_tag_t asn_DEF_mappingType_tags_3[] = {
	(ASN_TAG_CLASS_CONTEXT | (1 << 2)),
	(ASN_TAG_CLASS_UNIVERSAL | (10 << 2))
};
static /* Use -fall-defs-global to expose */
asn_TYPE_descriptor_t asn_DEF_mappingType_3 = {
	"mappingType",
	"mappingType",
	&asn_OP_NativeEnumerated,
	asn_DEF_mappingType_tags_3,
	sizeof(asn_DEF_mappingType_tags_3)
		/sizeof(asn_DEF_mappingType_tags_3[0]) - 1, /* 1 */
	asn_DEF_mappingType_tags_3,	/* Same as above */
	sizeof(asn_DEF_mappingType_tags_3)
		/sizeof(asn_DEF_mappingType_tags_3[0]), /* 2 */
	{ &asn_OER_type_mappingType_constr_3, &asn_PER_type_mappingType_constr_3, NativeEnumerated_constraint },
	0, 0,	/* Defined elsewhere */
	&asn_SPC_mappingType_specs_3	/* Additional specs */
};

asn_TYPE_member_t asn_MBR_PDSCH_TimeDomainResourceAllocation_1[] = {
	{ ATF_POINTER, 1, offsetof(struct PDSCH_TimeDomainResourceAllocation, k0),
		(ASN_TAG_CLASS_CONTEXT | (0 << 2)),
		-1,	/* IMPLICIT tag at current level */
		&asn_DEF_NativeInteger,
		0,
		{ &asn_OER_memb_k0_constr_2, &asn_PER_memb_k0_constr_2,  memb_k0_constraint_1 },
		0, 0, /* No default value */
		"k0"
		},
	{ ATF_NOFLAGS, 0, offsetof(struct PDSCH_TimeDomainResourceAllocation, mappingType),
		(ASN_TAG_CLASS_CONTEXT | (1 << 2)),
		-1,	/* IMPLICIT tag at current level */
		&asn_DEF_mappingType_3,
		0,
		{ 0, 0, 0 },
		0, 0, /* No default value */
		"mappingType"
		},
	{ ATF_NOFLAGS, 0, offsetof(struct PDSCH_TimeDomainResourceAllocation, startSymbolAndLength),
		(ASN_TAG_CLASS_CONTEXT | (2 << 2)),
		-1,	/* IMPLICIT tag at current level */
		&asn_DEF_NativeInteger,
		0,
		{ &asn_OER_memb_startSymbolAndLength_constr_6, &asn_PER_memb_startSymbolAndLength_constr_6,  memb_startSymbolAndLength_constraint_1 },
		0, 0, /* No default value */
		"startSymbolAndLength"
		},
};
static const int asn_MAP_PDSCH_TimeDomainResourceAllocation_oms_1[] = { 0 };
static const ber_tlv_tag_t asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1[] = {
	(ASN_TAG_CLASS_UNIVERSAL | (16 << 2))
};
static const asn_TYPE_tag2member_t asn_MAP_PDSCH_TimeDomainResourceAllocation_tag2el_1[] = {
    { (ASN_TAG_CLASS_CONTEXT | (0 << 2)), 0, 0, 0 }, /* k0 */
    { (ASN_TAG_CLASS_CONTEXT | (1 << 2)), 1, 0, 0 }, /* mappingType */
    { (ASN_TAG_CLASS_CONTEXT | (2 << 2)), 2, 0, 0 } /* startSymbolAndLength */
};
asn_SEQUENCE_specifics_t asn_SPC_PDSCH_TimeDomainResourceAllocation_specs_1 = {
	sizeof(struct PDSCH_TimeDomainResourceAllocation),
	offsetof(struct PDSCH_TimeDomainResourceAllocation, _asn_ctx),
	asn_MAP_PDSCH_TimeDomainResourceAllocation_tag2el_1,
	3,	/* Count of tags in the map */
	asn_MAP_PDSCH_TimeDomainResourceAllocation_oms_1,	/* Optional members */
	1, 0,	/* Root/Additions */
	-1,	/* First extension addition */
};
asn_TYPE_descriptor_t asn_DEF_PDSCH_TimeDomainResourceAllocation = {
	"PDSCH-TimeDomainResourceAllocation",
	"PDSCH-TimeDomainResourceAllocation",
	&asn_OP_SEQUENCE,
	asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1,
	sizeof(asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1)
		/sizeof(asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1[0]), /* 1 */
	asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1,	/* Same as above */
	sizeof(asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1)
		/sizeof(asn_DEF_PDSCH_TimeDomainResourceAllocation_tags_1[0]), /* 1 */
	{ 0, 0, SEQUENCE_constraint },
	asn_MBR_PDSCH_TimeDomainResourceAllocation_1,
	3,	/* Elements count */
	&asn_SPC_PDSCH_TimeDomainResourceAllocation_specs_1	/* Additional specs */
};

