// One-shot probe: writes the runtime subgroup_size into result[0].
// Dispatched once at Model::load to find out what the driver actually picked
// (Vulkan adapters report the device's subgroup_min..max range, not the size
// chosen for any given pipeline; RADV's `RADV_PERFTEST=cswave32` and
// `VK_EXT_subgroup_size_control` both shift the actual choice independently
// of that range).

@group(0) @binding(0) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size) sg_size: u32,
) {
  if (lid.x == 0u) {
    result[0] = sg_size;
  }
}
