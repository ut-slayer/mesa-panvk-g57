# PanVK on Mali-G57 (Valhall v9 + Job Manager): port report

**Provenance notice.** This report, and the driver code it describes, were written by an AI assistant (Claude, Anthropic)
at the request of the owner of the hardware, who directed, tested and reviewed the work. Per Mesa's AGENTS.md the code
has not been submitted to Mesa's GitLab and every commit carries `Generated-by: LLM`. It is shared for information:
anyone picking it up does so under their own responsibility. (biblioklept)

Repository: https://github.com/ut-slayer/mesa-panvk-g57 (this file is `PANVK-G57-REPORT.md` at its root). Branch `panvk-v9-26.3`, on top of `mesa-26.3.0-rc3`.
Dates: 2026-11-05 to 2026-11-07.

## 1. Summary

- PanVK builds, loads and runs on a Mali-G57 MC1 (GPU ID 0x9091, Allwinner A523, TrimUI Smart Pro S) with the
  `panfrost` kernel driver (Linux 6.18) and the Job Manager backend. Arch 9 = Valhall descriptors and shaders +
  Bifrost-style job chains, so the port reuses the existing Valhall (`PAN_ARCH >= 9`) code for state and the JM
  backend for submission, with a new draw path based on `MALLOC_VERTEX_JOB`.
- dEQP-VK 1.4.7.1 mustpass, all 253 top-level groups run: about 390,400 pass, no crashes, and the remaining
  failures are all accounted for (hardware waiver 4936, three known JM failures, two loader-related). The only cases
  not executed are ~137 `constant_data` shaders with 256 KiB of constants, which do not finish compiling (§4.12). Details in §5.
- vkmark 543 at 1280x720 (Wayland). PPSSPP in Vulkan: Soul Calibur Broken Destiny 60/60 fps, God of War Ghost of
  Sparta 30/30 fps (its 30 fps compat hack), no GPU faults.
- 33 commits, 25 files. 28 are the v9 port, one (`4b06370d193`, §4.11) is a compile-time fix in the shared Bifrost/Valhall
  compiler, and 4 (`74bd05136b0`, `44a77c514a8`, `f1e6216d5f7`, `ae2a2557bff`) are Bifrost/G52 performance work done in
  parallel on an RGB10 Max 3 and are not covered here.
- The compiler used is the C Bifrost/Valhall backend. Kraid was not built (no bindgen on the build host), so nothing
  below has been checked with Kraid.

## 2. Environment

| | |
|---|---|
| GPU | Mali-G57 MC1, 1 shader core, 150-888 MHz (devfreq `simple_ondemand`) |
| SoC / CPU | Allwinner A523, 8x Cortex-A55, 1 GiB RAM |
| Kernel | 6.18, `panfrost` 1.4 until 2026-11-07, then 1.6 (the 7.x `WB_MMAP`/coherency patches backported; v9 is Job Manager, not CSF, so `panthor` does not apply) |
| Userspace | glibc 2.41, Wayland (sway), Vulkan loader 1.4.317, Mesa 26.2.6 Gallium panfrost for GL ES |
| Build | Buildroot cross toolchain (GCC 15.3), `-Dvulkan-drivers=panfrost -Dplatforms=wayland`, debugoptimized with asserts, `-Dpanfrost-rust=false` |
| Tests | dEQP-VK 1.4.7.1 (`vulkan-cts-1.4.7.1`), run on the device in batches under systemd with a thermal watchdog |

## 3. What the port consists of

Commits in order (all `panvk:` / `panvk/jm:` / `pan/libpan:` / `pan/bi:` prefixed, one logical change each):

| Commit | Change |
|---|---|
| `0401fff386d` | Gate JM-only render state (FB BO list, needs_load/store, layer id) on `PAN_ARCH < 10` instead of `< 9`. |
| `6eed4723b31` | Make the Valhall frame shader path (preload/store) work with JM: TLS from the batch, "DCD flags 2" only on v10+, one DCD set shared by all layers. |
| `375dc0f4ea2` | Fill the Valhall tiler context fields (provoking vertex, layer) on v9, as Gallium `pan_jm.c` does. |
| `55e42ddafb7`, `3568d1c5c0f` | Port precompiled-kernel dispatch and compute dispatch to the v9 `COMPUTE_JOB` layout (payload + shader environment, resource table, push constants as FAU), following the CSF backend and `jm_launch_grid()`. |
| `ef1e88a95b9` | v9 draw path: Valhall descriptors as in the CSF backend, `MALLOC_VERTEX_JOB` as in `jm_emit_malloc_vertex_job()`. One job per draw and per layer. |
| `6eb71aa576b` | Build and expose arch 9 (meson `valhall_archs`/`jm_archs`, `panvk_arch_dispatch`, physical device; experimental, behind `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1`). |
| `491877ef211` | Set `vertex_array.packet` in the IDVS DCD (§4.1). |
| `ecf0e4c4d07`, `2f05076a032` | Indirect draws on v9: libpan helpers that patch the `MALLOC_VERTEX_JOB`, the per-instance attribute descriptors and the VS sysvals from the indirect buffer. |
| `60d38df71f1` | Cache ZSD, blend, DCD flags and push uniforms across draws (same dirty conditions as CSF): ~2.8 -> ~1.5 us CPU per draw. |
| `c0f8357985a`, `7001ff6e32d`, `735fc9d85ec` | Command buffer re-submission on JM (§4.2). |
| `b2a44984e9e` | Use the extended position FIFO whenever the VS writes point size or layer (§4.5). |
| `2b3c0702099` | gl_Layer and multiview on v9 handled like Bifrost (one job per layer, layer filtering through a sysval, gl_ViewIndex as sysval). |
| `cf4fffdd2be` | Split the batch past 1M vertices (§4.4). |
| `ac6387c66bb` | Accept timestamp query pools on JM (return 0; `timestampValidBits = 0`). PPSSPP creates them unconditionally. |
| `7ec95fcc82d` | Split the batch when the provoking vertex mode changes (it lives in the tiler context and FBD on JM). |
| `6275ffc63d5` | No resource waits for input attachment loads on v9 (§4.6). |
| `fe2621c0b05` | `VK_EXT_multi_draw` on JM (draw loop + `gl_DrawID` sysval). |
| `0f278d3b8fa` | Don't preallocate the 64 KiB descriptor pool per JM command buffer (`allocate_many_primary` creates 10,000 of them: OOM on 1 GiB). Applies to v6/v7 too. |
| `c7d67999e15` | Accept and advertise low global priority on JM (refusing a priority at or below medium is not allowed; the two `create_device_global_priority` cases are in `panfrost-g52-fails.txt`). |
| `7cf17e9d424` | Copy multisampled images through the graphics pipeline on v9 (§4.3). |
| `4259a29700e` | Serialize divergent buffer resource handles on v9 (§4.7). |
| `a004e93323f`, `d0915f2c725`, `f50c8e2a492` | Indirect draw fixes found by a code audit (§4.8). |

Decisions taken without a reference: "JM" is spelled `PAN_ARCH < 10` everywhere; layers are handled like v7 (one
tiler context and one job per layer) because Valhall hardware layer selection could not be made to work with JM.

## 4. Hardware and driver findings on v9 (the part likely to be useful upstream)

### 4.1 `vertex_array.packet` must be set in the IDVS DCD
With memory-allocated IDVS, every draw faulted with `DATA_INVALID_FAULT` until the DCD vertex array had the packet bit
set, as Gallium does. Nothing else in the DCD needs to differ from the CSF backend.

### 4.2 Re-submitting command buffers on JM
Gallium never re-issues jobs, so these only show up in Vulkan:
- The hardware writes the vertex array pointer and strides into the DCD of a `MALLOC_VERTEX_JOB`. A second submission
  of the same batch faults (`DATA_INVALID_FAULT`) unless they are restored (pointer 0, strides 0, packet bit set).
- The JM re-issue path (`panvk_queue_submit_batch`) patches job headers from the CPU. If the previous run of that batch
  is still executing (vkmark re-submits per-swapchain-image command buffers right after acquire), the jobs in flight
  are rewritten: 11-23 fps and hundreds of GPU faults on the G57. Fixed by waiting on a per-batch syncobj before
  patching (`7001ff6e32d`, `735fc9d85ec`). The same code path exists for v6/v7 in main; we did not check whether the
  G52 shows the problem.

### 4.3 Multisampled storage image stores (`LEA_TEX`) on the G57
vk_meta copies multisampled images with a compute shader that stores each sample through a multisampled storage image.
On the G57, with a 2D texture descriptor carrying a sample count, the sample coordinate of the store is ignored: only
sample 0 is written. The v10 CSF backend runs the same shader and descriptor and passes, so this looks like a
first-generation Valhall difference. Exposing the samples as the Z slices of a 3D descriptor (the layout the hardware
uses, and what Bifrost does) works for non-array images, but the hardware does not ignore the array size of a 3D
descriptor, so array images still break; setting a sample count on a 3D descriptor breaks addressing entirely. We
route multisampled copies through the graphics pipeline on v9 (sample-shaded `texelFetch` + tile buffer): 288
`api.copy_and_blit` cases fixed. Storage images with samples are not exposed to applications anyway.

### 4.4 Tiler heap exhaustion without incremental rendering
IDVS jobs take the vertex packets and polygon lists of the whole batch from the tiler heap (128 MiB) and only release
them at fragment time. kbase handles exhaustion with incremental rendering; `panfrost` has nothing equivalent, so a
batch with enough vertices (`memory.pipeline_barrier.*1048576`, a 1M-point test) fails with a tiler OOM and draws
nothing. The driver splits the batch past 1M vertices. A byte budget (vertices x packet stride) would be more accurate.

### 4.5 Extended position FIFO selection
v9 has no position FIFO format field; the extended FIFO (room for point size and layer) is selected through the point
size format. Enabling it only for point primitives meant a VS writing `gl_Layer` overwrote the next vertex's position
in vk_meta blits (7 of 384 `blit_image` cases passing). It must be enabled whenever the VS needs it, as CSF does.

### 4.6 Input attachment loads and `wait_resource`
PanVK loads read-only colour input attachments with `load_tile_res_pan` (wait only on the writer); that wait does not
exist on v9 and the compiler asserts in `va_insert_flow.c` (`!I->wait_resource || ctx->arch >= 10`). On v9 all input
attachment loads use the full wait.

### 4.7 Divergent buffer resource handles (`LD_PKA` / `LEA_PKA`)
On the G57, the buffer resource handle of `LD_PKA`/`LEA_PKA` is taken from a single lane of the warp. When invocations
select between pointers to different SSBOs (variable pointers, `OpSelect`/`OpPhi` of storage buffer pointers), lanes
whose handle differs from the one the hardware picked read and write the wrong buffer:
`spirv_assembly.instruction.compute.variable_pointers.compute.*_two_buffers` (8 cases) and
`spirv1p4.opselect.ssbo_pointers_2_select` fail with wrong data. The C backend and Kraid both emit per-lane handles
(`OpLdPka`/`OpLeaPka` in `kraid/nir.rs`), so Kraid would have the same problem on this hardware; whether v10 behaves
the same is unknown to us (these tests are not in `panfrost-g610-fails.txt`). The fix extends
`bi_lower_divergent_indirects` (already used for divergent attribute and image indices) to `load_ubo`, `load_ssbo`
and `load_ssbo_address` when the handle is divergent, on v9 only; shaders with uniform handles are unchanged.

### 4.8 Indirect draw helpers and re-submission (found by audit, not by CTS)
dEQP never re-submits a command buffer, so these were found by reviewing the code and confirmed with a test that
submits the same command buffer four times while toggling the indirect buffers:
- A draw whose indirect count was 0 once was turned into a `NULL` job by the helper and never turned back into
  `MALLOC_VERTEX` when a later submission had something to draw (`f50c8e2a492`). The re-issue path also read the job
  type back from GPU memory (which the helper rewrites) through a cached mapping to decide which DCDs to restore; it
  now keeps a CPU-side list of the IDVS DCDs.
- Per-instance attribute offsets were patched by adding `firstInstance * stride` to the descriptor in place, so they
  accumulated on every submission (`d0915f2c725`). **The v10 CSF helper uses the same load/add/store pattern; it may
  have the same re-submission bug.** Not verified on v10 hardware.
- The draw depended on its helper through the second dependency slot, which `pan_jc_add_job()` overrides with the
  previous tiling job of the chain; it worked only because the helper is always the job emitted right before, with
  prefetch suppressed. Now in the first slot (`a004e93323f`).

### 4.9 Things in main that look wrong for v9 but were not changed (please check)
- `panvk_vX_query_pool.c`: `reports_per_query = core_id_range` only for `PAN_ARCH < 9`; on v9 it stays 1. Gallium
  reserves one report per core on every architecture. Invisible on an MC1; on a multi-core v9 GPU (G57 MC2, G68, G77)
  cores 1..n would write past the report if occlusion counters are per core on JM.
- `panvk_vX_shader.c`: `allow_merging_workgroups` is enabled for `>= 9` but `mark_all_access_non_uniform` only for
  `>= 10`, so on v9 a small compute workgroup indexing descriptors with a per-workgroup value (without
  `nonuniformEXT`) can hit real divergence inside a merged subgroup.
- `jm/panvk_vX_cmd_draw.c`: the DCD flags dirty condition does not include logic op / blend constants, while
  `evaluate_per_sample` depends on `cb.info.needs_shader`.
- Global priority on JM never reaches the kernel (`jm_ctx_handle = 0`); Gallium main creates a JM context with a
  priority, and its comment notes that without an explicit context nothing kills in-flight jobs on close. We see the
  corresponding symptom: killing a process with jobs in flight sometimes leaves a page fault in device-private shader
  memory and a `gpu sched timeout` (also on v7 and with upstream Mesa).

### 4.10 Quadratic loop-body computation in the Bifrost/Valhall compiler (upstream too)
`find_all_dominated_predecessors()` in `bifrost_compile.c` (used by `bi_find_loop_blocks()` for the spill cost and in
the SSA spiller) iterates `num_blocks - 1` times over every block, for every loop header. The compute shaders of
`dEQP-VK.subgroups.clustered.*` have hundreds of blocks: compiling the 7 pipelines of one case took 25 s of CPU on a
Cortex-A55, 45 % of it in this function (profiled with a SIGPROF sampler preloaded together with the driver, shader
disk cache disabled). `4b06370d193` replaces it with a backward walk over predecessor edges inside the dominated
set (linear, same result): 15 s per case, the rest being NIR validation (asserts build) and NIR dominance. Mesa main
has the same loop; the change is not v9-specific.

### 4.11 `constant_data` cases with 256 KiB of constants never finish compiling (core NIR, not panfrost)
`dEQP-VK.spirv_assembly.constant_data.*` with `count`/`actual` = 262128 / width (32,766 to 262,128 constants) hit the
dEQP watchdog; one case run without the watchdog did not finish in 15 min. 99.9 % of the time is in
`nir_opt_copy_prop_vars` (`lookup_entry_and_kill_aliases_copy_array` -> `nir_compare_derefs_and_paths`): each
`store_deref` to an element of the private array is compared against every earlier entry of the same variable,
quadratic in the number of constants. PanVK calls the pass in `panvk_vX_shader.c` and in `bi_optimize_loop()`. Not a
wrong result; left as is (~137 cases not executed).

### 4.12 Not Mesa, but relevant to anyone testing on this SoC
The Allwinner A523 GPU clock as set up by the vendor-derived device tree ran the 648-840 MHz OPPs by pulse-skipping
`pll-gpu` at 888 MHz, and jumping between `pll-periph0` and `pll-gpu` corrupted bits and produced GPU faults. 888 MHz
fixed or <= 600 MHz were fine. Fixed on the kernel side (exact PLL programming); the symptoms looked exactly like
driver bugs until isolated with a clock sweep.

## 5. dEQP-VK results (1.4.7.1 mustpass)

Full run of the mustpass by top-level group on the device, all 253 groups, with a 2400 s budget per group (3 h for the
five large `subgroups.*` groups, 10 h for `renderpasses.dynamic_rendering`) and a 300 s per-case watchdog:

| | Cases |
|---|---|
| Pass | ~390,400 |
| Not supported | ~609,000 (features PanVK does not advertise on v9; includes the whole `subgroups.partitioned` group, `VK_NV_shader_subgroup_partitioned`) |
| Fail | ~506, all classified below |
| Crash / device lost | 0 |
| Not executed | ~137 of `spirv_assembly.constant_data.*` (§4.11) |

Per-group highlights (pass / fail): `api.copy_and_blit` 93,371 / 0; `pipeline.monolithic` 26,794 / 0;
`draw.renderpass` 1,716 / 0; `draw.dynamic_rendering` 2,839 / 0; `renderpasses.dynamic_rendering` 16,551 / 0;
`memory` 5,480 / 0 (with panfrost 1.6; 3 failures with 1.4, two `pipeline_barrier.host_write_*.1048576` and one sporadic
dma_buf AFBC case, all gone with the newer kernel driver); `subgroups.*` (19 groups) 5,845 / 0;
`spirv_assembly.instruction` 15,792 / 0; `spirv_assembly.constant_data` 1,424 / 0; `glsl.*` all pass except the two groups below.

Remaining failures, all understood:

| Cases | Group | Cause |
|---|---|---|
| 500 | `glsl.texture_functions.*offset*.mirrored_repeat.*` | Hardware; Khronos waiver 4936 for Arm, G57 (0x9091) listed; skipped in `panvk-skips.txt`. |
| 3 | `rasterization.primitives.no_stipple.lines*` | Known JM failure, in `panfrost-g52-fails.txt`; same line setup as v10, where it passes. Not investigated further. |
| 1 | `glsl.loops.special.do_while_dynamic_iterations.dowhile_trap_vertex` | Known, in the G610 and G720 fail lists (GPU hang until the scheduler timeout). |
| 1 | `api.version_check.entry_points` | Loader 1.4.317 predates `vkReleaseSwapchainImagesKHR`, `vkCmdEndRendering2KHR`, `VK_EXT_present_timing`. |
| 1 | `api.driver_properties.conformance_version` | 0.0.0.0 advertised for non-v10, as expected for an uncertified driver. |
| 2 (ResourceError) | `api.object_management.max_concurrent.device{,_group}` | 32 devices x tiler heap on 1 GiB; known JM failure (G52 list). |
| ~137 not run | `spirv_assembly.constant_data.*{count,actual}_{32766..262128}_*` | Compile time: quadratic `nir_opt_copy_prop_vars` (§4.11); not a wrong result. Not in Mesa CI. |

## 6. Performance

| | |
|---|---|
| CPU per draw (release build, microbenchmark) | ~1.5 us after caching ZSD/blend/DCD flags/push uniforms (2.8 before) |
| vkmark, 1280x720, Wayland | 543 (first working build: 132) |
| PPSSPP Vulkan | Soul Calibur Broken Destiny 60/60 fps; God of War Ghost of Sparta 30/30 fps (its 30 fps hack); 0 GPU faults |
| For reference, GL ES on the same device | glmark2 546-574 with Gallium panfrost |

Batch splits are the main remaining GPU-side cost: every `vkCmdPipelineBarrier` and every `vkCmdDispatch` inside a
render pass, EndQuery, the 1M-vertex limit and provoking vertex changes all end the batch (two submits, L2 flush,
framebuffer spill and reload). Chaining dispatches inside a batch with per-job local storage (as `jm_launch_grid`)
and a byte-based tiler heap budget are the obvious next steps; not done.

## 7. Not done / open

- Kraid: not built (bindgen missing on the build host); everything above uses the C backend.
- Occlusion query reports per core on JM v9 (§4.9), `mark_all_access_non_uniform` gating, DCD flags dirty bits.
- `VK_KHR_display` (KMS): PanVK passes -1 as the display fd to the WSI; a change is written but untested.
- `no_stipple.lines` on JM, `max_concurrent.device`, `constant_data` compile time (§4.11).
- About 40 features and extensions are gated on `PAN_ARCH >= 10` in `panvk_vX_physical_device.c` (and Vulkan 1.4);
  many were simply never enabled for JM. Not reviewed one by one yet.
- Nothing has been tested on a multi-core v9 GPU.

## 8. How to reproduce

```
meson setup build -Dvulkan-drivers=panfrost -Dgallium-drivers= -Dplatforms=wayland \
      -Dpanfrost-rust=false -Dbuildtype=debugoptimized -Db_ndebug=false
PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1 VK_ICD_FILENAMES=.../panfrost_icd.json vulkaninfo --summary
```
`PANVK_DEBUG=trace PANDECODE_DUMP_FILE=/tmp/pv.dump` dumps every submission. The device-side test programs
(compute, direct/indirect draws with four re-submissions, point count, per-draw timing) are small standalone C
programs outside the Mesa tree and can be provided on request.
