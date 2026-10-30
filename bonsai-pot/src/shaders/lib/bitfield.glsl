// Bitfield extraction via direct SPIR-V intrinsics.

// OpBitFieldUExtract(Base, Offset, Count): extract `count` bits at `offset`,
// zero-filled. Result type matches `Base` (uint).
spirv_instruction(id = 203) uint extract_bits(uint base, uint offset, uint count);

// OpBitFieldSExtract(Base, Offset, Count): same, sign-extended from the high
// bit of the extracted field. Result type matches `Base` (int).
spirv_instruction(id = 202) int extract_bits_signed(int base, uint offset, uint count);
