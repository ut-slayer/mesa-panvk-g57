/*
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdatomic.h>
#include <stdint.h>
#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"

#pragma once

#if (PAN_ARCH == 6 || PAN_ARCH == 7)
/* Keep in sync with panvk_varying_buf_id */
enum panlib_varying_buf_id {
   PANLIB_VARY_BUF_GENERAL,
   PANLIB_VARY_BUF_POSITION,
   PANLIB_VARY_BUF_PSIZ,

   /* Keep last */
   PANLIB_VARY_BUF_MAX,
};

struct libpan_draw_helper_index_min_max_result {
    uint32_t min;
    uint32_t max;
};

struct libpan_draw_helper_attrib_buf_info {
    uint32_t divisor;
    uint32_t stride;
    bool per_instance;
};

struct libpan_draw_helper_attrib_info {
    uint32_t base_offset;

    /* When the attribute is per instance, the stride otherwise 0 */
    uint32_t stride;
};

/* A direct indexed draw patched by panlib_draw_indexed_direct_group_helper.
 * Pointers are GPU addresses, 0 when unused. */
struct libpan_draw_helper_indexed_direct_draw {
    uint64_t indices;
    uint64_t varying_bufs_descs;
    uint64_t raw_vertex_offset_sysval;
    uint64_t idvs_job;
    uint64_t vertex_job;
    uint64_t tiler_job;
    int32_t vertex_offset;
    uint32_t index_count;
    uint32_t varying_buf_mask;
    uint8_t patch_psiz;
    uint8_t index_bytes_log2;
    uint8_t primitive_restart;
    uint8_t pad;
};

/* Header of the draw array of a group helper: draws follow it. */
struct libpan_draw_helper_indexed_direct_group {
    uint32_t count;
    uint32_t pad;
};
#endif
