/*
 * Copyright © 2024 Collabora Ltd.
 * Copyright © 2026 NXP
 *
 * Derived from tu_cmd_buffer.c which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_draw.h"
#include "panvk_cmd_meta.h"
#include "panvk_cmd_precomp.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_meta.h"
#include "panvk_priv_bo.h"
#include "panvk_shader.h"

#include "draw_helper.h"
#include "pan_desc.h"
#include "pan_earlyzs.h"
#include "pan_encoder.h"
#include "pan_format.h"
#include "pan_jc.h"
#include "pan_props.h"
#include "pan_shader.h"

#include "vk_format.h"
#include "vk_meta.h"
#include "vk_pipeline_layout.h"

struct panvk_draw_data {
   struct panvk_draw_info info;
#if PAN_ARCH < 9
   unsigned vertex_range;
   unsigned padded_vertex_count;
   struct mali_invocation_packed invocation;
   struct {
      uint64_t varyings;
      uint64_t attributes;
      uint64_t attribute_bufs;
   } vs;
   struct {
      uint64_t rsd;
      uint64_t varyings;
   } fs;
   uint64_t varying_bufs;
   uint64_t position;
   union {
      uint64_t psiz;
      float line_width;
   };
#endif
   uint64_t tls;
   uint64_t fb;
   const struct pan_tiler_context *tiler_ctx;
#if PAN_ARCH < 9
   uint64_t viewport;
   struct {
      struct pan_ptr vertex_copy_desc;
      struct pan_ptr frag_copy_desc;
      union {
         struct {
            struct pan_ptr vertex;
            struct pan_ptr tiler;
         };
         struct pan_ptr idvs;
      };
   } jobs;
   struct {
      uint64_t attribs;
      uint64_t attrib_bufs;
   } indirect_info;
#else
   /* State shared by all the per-layer jobs of a draw. */
   uint64_t zsd;
   uint64_t blend;
   uint32_t blend_count;
   struct mali_dcd_flags_0_packed dcd_flags_0;
   struct mali_dcd_flags_1_packed dcd_flags_1;
   struct mali_scissor_packed scissor;
   float z_min, z_max;
   struct pan_ptr job;
#endif
};

static bool
is_indirect_draw(const struct panvk_draw_data *draw)
{
   return draw->info.indirect.buffer_dev_addr != 0 ||
          draw->info.index.index_size != 0;
}

static bool
has_depth_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_DEPTH_BIT) != 0;
}

static bool
has_stencil_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_STENCIL_BIT) != 0;
}

static bool
writes_depth(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_depth_att(cmdbuf) && ds->depth.test_enable &&
          ds->depth.write_enable && ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

static bool
writes_stencil(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_stencil_att(cmdbuf) && ds->stencil.test_enable &&
          ((ds->stencil.front.write_mask &&
            (ds->stencil.front.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.depth_fail != VK_STENCIL_OP_KEEP)) ||
           (ds->stencil.back.write_mask &&
            (ds->stencil.back.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.depth_fail != VK_STENCIL_OP_KEEP)));
}

static bool
ds_test_always_passes(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   if (!has_depth_att(cmdbuf))
      return true;

   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;

   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;

   return true;
}

static inline enum mali_func
translate_compare_func(VkCompareOp comp)
{
   STATIC_ASSERT(VK_COMPARE_OP_NEVER == (VkCompareOp)MALI_FUNC_NEVER);
   STATIC_ASSERT(VK_COMPARE_OP_LESS == (VkCompareOp)MALI_FUNC_LESS);
   STATIC_ASSERT(VK_COMPARE_OP_EQUAL == (VkCompareOp)MALI_FUNC_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_LESS_OR_EQUAL == (VkCompareOp)MALI_FUNC_LEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER == (VkCompareOp)MALI_FUNC_GREATER);
   STATIC_ASSERT(VK_COMPARE_OP_NOT_EQUAL == (VkCompareOp)MALI_FUNC_NOT_EQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 (VkCompareOp)MALI_FUNC_GEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_ALWAYS == (VkCompareOp)MALI_FUNC_ALWAYS);

   return (enum mali_func)comp;
}

static enum mali_stencil_op
translate_stencil_op(VkStencilOp in)
{
   switch (in) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("Invalid stencil op");
   }
}

static enum mali_draw_mode
translate_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
#if PAN_ARCH >= 9
   case MESA_PRIM_LINES_ADJACENCY:
      return MALI_DRAW_MODE_LINES_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_LINE_STRIP_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLES_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLE_STRIP_ADJACENCY;
#endif
   default:
      UNREACHABLE("Invalid primitive type");
   }
}

static VkResult
panvk_draw_prepare_tiler_context(struct panvk_cmd_buffer *cmdbuf,
                                 struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   VkResult result =
      panvk_per_arch(cmd_prepare_tiler_context)(cmdbuf, draw->info.layer_id);
   if (result != VK_SUCCESS)
      return result;

   draw->tiler_ctx = &batch->tiler.ctx;
   return VK_SUCCESS;
}

static void
set_provoking_vertex_mode(struct panvk_cmd_buffer *cmdbuf,
                          enum u_tristate first_provoking_vertex)
{
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;

   if (first_provoking_vertex != U_TRISTATE_UNSET) {
      /* The provoking vertex mode lives in the tiler context and the FBD,
       * which are per batch. If the application switches modes within a
       * render pass (PPSSPP does), split the batch once its tiler context
       * has been emitted with the previous mode. The split spills and
       * reloads the framebuffer. */
      if (state->render.first_provoking_vertex != U_TRISTATE_UNSET &&
          state->render.first_provoking_vertex != first_provoking_vertex &&
          cmdbuf->cur_batch && cmdbuf->cur_batch->tiler.ctx_descs.gpu) {
         panvk_per_arch(cmd_close_batch)(cmdbuf);
         panvk_per_arch(cmd_open_batch)(cmdbuf);
      }
      state->render.first_provoking_vertex = first_provoking_vertex;
   }

   /* Once we emit the first FBDs/TDs, we need to commit to a state. If we
    * choose the wrong one, we will fail the assert when the next application
    * draw happens (with a different state). Use PROVOKING_VERTEX_MODE_FIRST
    * because it's the vulkan default, and so likely to be right more often.
    *
    * TODO: handle this case better */
   if (state->render.first_provoking_vertex == U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = U_TRISTATE_YES;
}

#if PAN_ARCH < 9
static VkResult
panvk_draw_prepare_fs_rsd(struct panvk_cmd_buffer *cmdbuf,
                          struct panvk_draw_data *draw)
{
   bool dirty = dyn_gfx_state_dirty(cmdbuf, RS_RASTERIZER_DISCARD_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_FACTORS) ||
                dyn_gfx_state_dirty(cmdbuf, RS_LINE_MODE) ||
                /* line mode needs primitive topology */
                dyn_gfx_state_dirty(cmdbuf, IA_PRIMITIVE_TOPOLOGY) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP) ||
                dyn_gfx_state_dirty(cmdbuf, CB_ATTACHMENT_COUNT) ||
                dyn_gfx_state_dirty(cmdbuf, CB_COLOR_WRITE_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
                dyn_gfx_state_dirty(cmdbuf, CB_BLEND_CONSTANTS) ||
                dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_COMPARE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_REFERENCE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_RASTERIZATION_SAMPLES) ||
                dyn_gfx_state_dirty(cmdbuf, MS_SAMPLE_MASK) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_COVERAGE_ENABLE) ||
                dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
                gfx_state_dirty(cmdbuf, FS) || gfx_state_dirty(cmdbuf, OQ) ||
                gfx_state_dirty(cmdbuf, RENDER_STATE);

   if (!dirty) {
      draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
      return VK_SUCCESS;
   }

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;
   const struct vk_depth_stencil_state *ds = &dyns->ds;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct pan_shader_info *fs_info = fs ? &fs->info : NULL;
   uint32_t bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
   bool test_s = has_stencil_att(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att(cmdbuf) && ds->depth.test_enable;
   bool writes_z = writes_depth(cmdbuf);
   bool writes_s = writes_stencil(cmdbuf);

   bool msaa = dyns->ms.rasterization_samples > 1;
   if ((ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
        ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) &&
       rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM) {
      /* we need to disable MSAA when rendering bresenham lines.
       *
       * From the Vulkan spec:
       *   "When Bresenham lines are being rasterized, sample locations may
       *    all be treated as being at the pixel center (this may affect
       *    attribute and depth interpolation).""
       */
      msaa = false;
   }

   struct pan_ptr ptr = panvk_cmd_alloc_desc_aggregate(
      cmdbuf, PAN_DESC(RENDERER_STATE), PAN_DESC_ARRAY(bd_count, BLEND));
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_renderer_state_packed *rsd = ptr.cpu;
   struct mali_blend_packed *bds = ptr.cpu + pan_size(RENDERER_STATE);
   struct panvk_blend_info *binfo = &cmdbuf->state.gfx.cb.info;

   uint64_t fs_code = panvk_shader_variant_get_dev_addr(fs);

   if (fs_info != NULL) {
      panvk_per_arch(blend_emit_descs)(cmdbuf, bds);
   } else {
      for (unsigned i = 0; i < bd_count; i++) {
         pan_pack(&bds[i], BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
         }
      }
   }

   pan_pack(rsd, RENDERER_STATE, cfg) {
      bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;

      if (fs) {
         pan_shader_prepare_rsd(fs_info, fs_code, &cfg);

         uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                           MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
         uint8_t rt_written = color_attachment_written_mask(
            fs, &cmdbuf->vk.dynamic_graphics_state.cal);
         uint8_t rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
         enum pan_earlyzs_zs_tilebuf_read zs_read =
            (z_attachment_read(fs, &dyns->ial) ||
             s_attachment_read(fs, &dyns->ial))
               ? PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT
               : PAN_EARLYZS_ZS_TILEBUF_NOT_READ;

         cfg.properties.allow_forward_pixel_to_kill =
            fs_info->fs.can_fpk && !(rt_mask & ~rt_written) &&
            !(rt_read & rt_written) && !alpha_to_coverage &&
            !binfo->any_dest_read;

         bool writes_zs = writes_z || writes_s;
         bool zs_always_passes = ds_test_always_passes(cmdbuf);
         bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                   MALI_OCCLUSION_MODE_DISABLED;

         struct pan_earlyzs_state earlyzs =
            pan_earlyzs_get(fs->fs.earlyzs_lut, writes_zs || oq,
                            alpha_to_coverage, zs_always_passes, zs_read);

         /* early ZS check for FPK is performed by HW on v7+ */
         cfg.properties.allow_forward_pixel_to_be_killed =
            !fs->info.writes_global &&
            ((PAN_ARCH > 6) || earlyzs.kill != MALI_PIXEL_KILL_FORCE_LATE);

         cfg.properties.pixel_kill_operation = earlyzs.kill;
         cfg.properties.zs_update_operation = earlyzs.update;
         cfg.multisample_misc.evaluate_per_sample =
            (fs->info.fs.sample_shading && dyns->ms.rasterization_samples > 1);
      } else {
         cfg.properties.depth_source = MALI_DEPTH_SOURCE_FIXED_FUNCTION;
         cfg.properties.allow_forward_pixel_to_kill = true;
         cfg.properties.allow_forward_pixel_to_be_killed = true;
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      }

      cfg.multisample_misc.multisample_enable = msaa;
      cfg.multisample_misc.sample_mask = dyns->ms.sample_mask;

      cfg.multisample_misc.depth_function =
         test_z ? translate_compare_func(ds->depth.compare_op)
                : MALI_FUNC_ALWAYS;

      cfg.multisample_misc.depth_write_mask = writes_z;
      cfg.multisample_misc.fixed_function_near_discard =
      cfg.multisample_misc.fixed_function_far_discard =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.multisample_misc.fixed_function_depth_range_fixed =
         !rs->depth_clamp_enable;
      cfg.multisample_misc.shader_depth_range_fixed = true;

      cfg.stencil_mask_misc.stencil_enable = test_s;
      cfg.stencil_mask_misc.alpha_to_coverage = alpha_to_coverage;
      cfg.stencil_mask_misc.alpha_test_compare_function = MALI_FUNC_ALWAYS;
      cfg.stencil_mask_misc.front_facing_depth_bias = rs->depth_bias.enable;
      cfg.stencil_mask_misc.back_facing_depth_bias = rs->depth_bias.enable;

      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.stencil_mask_misc.aligned_line_ends = true;

      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;

      cfg.stencil_front.mask = ds->stencil.front.compare_mask;
      cfg.stencil_back.mask = ds->stencil.back.compare_mask;

      cfg.stencil_mask_misc.stencil_mask_front = ds->stencil.front.write_mask;
      cfg.stencil_mask_misc.stencil_mask_back = ds->stencil.back.write_mask;

      cfg.stencil_front.reference_value = ds->stencil.front.reference;
      cfg.stencil_back.reference_value = ds->stencil.back.reference;

      if (test_s) {
         cfg.stencil_front.compare_function =
            translate_compare_func(ds->stencil.front.op.compare);
         cfg.stencil_front.stencil_fail =
            translate_stencil_op(ds->stencil.front.op.fail);
         cfg.stencil_front.depth_fail =
            translate_stencil_op(ds->stencil.front.op.depth_fail);
         cfg.stencil_front.depth_pass =
            translate_stencil_op(ds->stencil.front.op.pass);
         cfg.stencil_back.compare_function =
            translate_compare_func(ds->stencil.back.op.compare);
         cfg.stencil_back.stencil_fail =
            translate_stencil_op(ds->stencil.back.op.fail);
         cfg.stencil_back.depth_fail =
            translate_stencil_op(ds->stencil.back.op.depth_fail);
         cfg.stencil_back.depth_pass =
            translate_stencil_op(ds->stencil.back.op.pass);
      }
   }

   cmdbuf->state.gfx.fs.rsd = ptr.gpu;
   draw->fs.rsd = cmdbuf->state.gfx.fs.rsd;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_varyings(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_link *link = &cmdbuf->state.gfx.link;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, PANVK_VARY_BUF_MAX + 1, ATTRIBUTE_BUFFER);
   if (!bufs.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct mali_attribute_buffer_packed *buf_descs = bufs.cpu;
   const struct vk_input_assembly_state *ia =
      &cmdbuf->vk.dynamic_graphics_state.ia;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   uint64_t psiz_buf = 0;

   for (unsigned i = 0; i < PANVK_VARY_BUF_MAX; i++) {
      uint32_t buf_size;
      uint64_t buf_addr;
      if (is_indirect_draw(draw)) {
         buf_addr = dev->poly_heap.buffer->addr.dev;
         buf_size = 0;
      } else {
         buf_size = draw->padded_vertex_count * draw->info.instance.count *
                    link->buf_strides[i];
         buf_addr =
            buf_size
               ? panvk_cmd_alloc_dev_mem(cmdbuf, varying, buf_size, 64).gpu
               : 0;

         if (buf_size && !buf_addr)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }

      pan_pack(&buf_descs[i], ATTRIBUTE_BUFFER, cfg) {
         cfg.stride = link->buf_strides[i];
         cfg.size = buf_size;
         cfg.pointer = buf_addr;
      }

      if (i == PANVK_VARY_BUF_POSITION)
         draw->position = buf_addr;

      if (i == PANVK_VARY_BUF_PSIZ)
         psiz_buf = buf_addr;
   }

   /* We need an empty entry to stop prefetching on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * PANVK_VARY_BUF_MAX), 0,
          pan_size(ATTRIBUTE_BUFFER));

   if (writes_point_size)
      draw->psiz = psiz_buf;
   else if (ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
            ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
      draw->line_width = cmdbuf->vk.dynamic_graphics_state.rs.line.width;
   else
      draw->line_width = 1.0f;

   draw->varying_bufs = bufs.gpu;
   draw->vs.varyings = panvk_priv_mem_dev_addr(link->vs.attribs);
   draw->fs.varyings = panvk_priv_mem_dev_addr(link->fs.attribs);
   return VK_SUCCESS;
}

static void
panvk_draw_emit_attrib_buf(
   const struct panvk_draw_data *draw,
   const struct vk_vertex_binding_state *buf_info, uint32_t stride,
   const struct panvk_attrib_buf *buf,
   struct mali_attribute_buffer_packed *desc,
   struct libpan_draw_helper_attrib_buf_info *helper_buf_info)
{
   uint64_t addr = buf->address & ~63ULL;
   unsigned size = buf->size + (buf->address & 63);
   unsigned divisor = draw->padded_vertex_count * buf_info->divisor;
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   struct mali_attribute_buffer_packed *buf_ext = &desc[1];

   /* In case of indirect draw, the descriptor will be patched at runtime */
   if (helper_buf_info != NULL) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.pointer = addr;
         cfg.size = size;
      }

      helper_buf_info->divisor = buf_info->divisor;
      helper_buf_info->stride = stride;
      helper_buf_info->per_instance = per_instance;
   } else if (draw->info.instance.count <= 1) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = per_instance ? 0 : stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!per_instance) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = draw->padded_vertex_count;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (!divisor) {
      /* instance_divisor == 0 means all instances share the same value.
       * Make it a 1D array with a zero stride.
       */
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = addr;
         cfg.size = size;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      unsigned divisor_r = 0, divisor_e = 0;
      unsigned divisor_d =
         pan_compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(desc, ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = addr;
         cfg.size = size;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }

      pan_cast_and_pack(buf_ext, ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = buf_info->divisor;
      }

      buf_ext = NULL;
   }

   /* If the buffer extension wasn't used, memset(0) */
   if (buf_ext)
      memset(buf_ext, 0, pan_size(ATTRIBUTE_BUFFER));
}

static void
panvk_draw_emit_attrib(const struct panvk_draw_data *draw,
                       const struct vk_vertex_attribute_state *attrib_info,
                       const struct vk_vertex_binding_state *buf_info,
                       const struct panvk_attrib_buf *buf,
                       struct mali_attribute_packed *desc,
                       struct libpan_draw_helper_attrib_info *helper_attrib_info)
{
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.buffer_index = buf_idx * 2;
      cfg.offset_enable = true;
      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;

      uint32_t offset = attrib_info->offset + (buf->address & 63);

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (helper_attrib_info != NULL) {
         helper_attrib_info->base_offset = offset;
         helper_attrib_info->stride = per_instance ? buf_info->stride : 0;
      } else {
         cfg.offset = offset;
         if (per_instance)
            cfg.offset += draw->info.instance.base * buf_info->stride;
      }
   }
}

/* Don't re-use vertex attributes, always re-emit them.  Descriptors depend on
 * per-draw parameters (e.g. vertex_count, instancing), which are likely to
 * change every draw, so don't bother trying to save an attribute[_buffer]
 * re-emission.
 */
static VkResult
panvk_draw_prepare_vs_attribs(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct panvk_shader_variant *vs_variant =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   unsigned num_imgs = vs->desc_info.others.count[PANVK_BIFROST_DESC_TABLE_IMG];
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   unsigned num_vs_attribs =
      num_imgs ? MAX_VS_ATTRIBS
               : MAX2(util_last_bit(vi->attributes_valid),
                      vs_variant->info.attribute_count);
   unsigned attrib_count = num_vs_attribs + num_imgs;

   unsigned attrib_buf_count = (num_vbs + num_imgs) * 2;
   struct pan_ptr bufs = panvk_cmd_alloc_desc_array(
      cmdbuf, attrib_buf_count + 1, ATTRIBUTE_BUFFER);
   struct mali_attribute_buffer_packed *attrib_buf_descs = bufs.cpu;
   struct pan_ptr attribs =
      panvk_cmd_alloc_desc_array(cmdbuf, attrib_count, ATTRIBUTE);
   struct mali_attribute_packed *attrib_descs = attribs.cpu;

   if (!bufs.gpu || (attrib_count && !attribs.gpu))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct libpan_draw_helper_attrib_buf_info *bufs_infos = NULL;
   struct libpan_draw_helper_attrib_info *attribs_infos = NULL;

   if (is_indirect_draw(draw)) {
      struct pan_ptr bufs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vbs * sizeof(struct libpan_draw_helper_attrib_buf_info), 8);
      struct pan_ptr attribs_infos_storage = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         num_vs_attribs * sizeof(struct libpan_draw_helper_attrib_info), 8);

      if (!bufs_infos_storage.gpu ||
          (num_vs_attribs && !attribs_infos_storage.gpu))
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos = bufs_infos_storage.gpu;
      cmdbuf->state.gfx.vs.indirect_attribs_infos = attribs_infos_storage.gpu;
      bufs_infos = bufs_infos_storage.cpu;
      attribs_infos = attribs_infos_storage.cpu;
   }

   for (unsigned i = 0; i < num_vbs; i++) {
      if (vi->bindings_valid & BITFIELD_BIT(i)) {
         struct libpan_draw_helper_attrib_buf_info *helper_buf_info =
            bufs_infos ? &bufs_infos[i] : NULL;
         panvk_draw_emit_attrib_buf(draw, &vi->bindings[i],
                                    dyns->vi_binding_strides[i],
                                    &cmdbuf->state.gfx.vb.bufs[i],
                                    &attrib_buf_descs[i * 2], helper_buf_info);
      } else {
         memset(&attrib_buf_descs[i * 2], 0, sizeof(*attrib_buf_descs) * 2);
      }
   }

   for (unsigned i = 0; i < num_vs_attribs; i++) {
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         unsigned buf_idx = vi->attributes[i].binding;
         struct libpan_draw_helper_attrib_info *helper_attrib_info =
            attribs_infos ? &attribs_infos[i] : NULL;
         panvk_draw_emit_attrib(draw, &vi->attributes[i],
                                &vi->bindings[buf_idx],
                                &cmdbuf->state.gfx.vb.bufs[buf_idx],
                                &attrib_descs[i], helper_attrib_info);
      } else {
         /* Unbound attributes read the zero default */
         pan_pack(&attrib_descs[i], ATTRIBUTE, cfg)
            cfg.format = MALI_PACK_FMT(CONSTANT, 0000, L);
      }
   }

   /* A NULL entry is needed to stop prefecting on Bifrost */
   memset(bufs.cpu + (pan_size(ATTRIBUTE_BUFFER) * attrib_buf_count), 0,
          pan_size(ATTRIBUTE_BUFFER));

   cmdbuf->state.gfx.vs.attrib_bufs = bufs.gpu;
   cmdbuf->state.gfx.vs.attribs = attribs.gpu;

   if (num_imgs) {
      cmdbuf->state.gfx.vs.desc.img_attrib_table =
         attribs.gpu + (MAX_VS_ATTRIBS * pan_size(ATTRIBUTE));
      cmdbuf->state.gfx.vs.desc.tables[PANVK_BIFROST_DESC_TABLE_IMG] =
         bufs.gpu + (num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2);
   }

   return VK_SUCCESS;
}

static void
panvk_draw_prepare_attributes(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   panvk_draw_prepare_vs_attribs(cmdbuf, draw);
   draw->vs.attributes = cmdbuf->state.gfx.vs.attribs;
   draw->vs.attribute_bufs = cmdbuf->state.gfx.vs.attrib_bufs;
   draw->indirect_info.attribs = cmdbuf->state.gfx.vs.indirect_attribs_infos;
   draw->indirect_info.attrib_bufs =
      cmdbuf->state.gfx.vs.indirect_attrib_bufs_infos;
}

static void
panvk_emit_viewport(struct panvk_cmd_buffer *cmdbuf,
                    struct mali_viewport_packed *vpd)
{
   const struct vk_viewport_state *vp = &cmdbuf->vk.dynamic_graphics_state.vp;

   if (vp->viewport_count < 1)
      return;

   const VkViewport *viewport = &vp->viewports[0];
   const VkRect2D *scissor = &vp->scissors[0];
   float minz, maxz;
   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                     &minz, &maxz);

   /* The spec says "width must be greater than 0.0" */
   assert(viewport->width >= 0);
   int minx = (int)viewport->x;
   int maxx = (int)(viewport->x + viewport->width);

   /* Viewport height can be negative */
   int miny = MIN2((int)viewport->y, (int)(viewport->y + viewport->height));
   int maxy = MAX2((int)viewport->y, (int)(viewport->y + viewport->height));

   assert(scissor->offset.x >= 0 && scissor->offset.y >= 0);
   minx = MAX2(scissor->offset.x, minx);
   miny = MAX2(scissor->offset.y, miny);
   maxx = MIN2(scissor->offset.x + scissor->extent.width, maxx);
   maxy = MIN2(scissor->offset.y + scissor->extent.height, maxy);

   /* Make sure we don't end up with a max < min when width/height is 0 */
   maxx = maxx > minx ? maxx - 1 : maxx;
   maxy = maxy > miny ? maxy - 1 : maxy;

   /* Clamp viewport scissor to valid range */
   minx = CLAMP(minx, 0, UINT16_MAX);
   maxx = CLAMP(maxx, 0, UINT16_MAX);
   miny = CLAMP(miny, 0, UINT16_MAX);
   maxy = CLAMP(maxy, 0, UINT16_MAX);

   pan_pack(vpd, VIEWPORT, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
      cfg.minimum_z = minz;
      cfg.maximum_z = maxz;
   }
}

static VkResult
panvk_draw_prepare_viewport(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   /* When rasterizerDiscardEnable is active, it is allowed to have viewport and
    * scissor disabled.
    * As a result, we define an empty one.
    */
   if (!cmdbuf->state.gfx.vpd || dyn_gfx_state_dirty(cmdbuf, VP_VIEWPORTS) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_SCISSORS) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
       dyn_gfx_state_dirty(cmdbuf, VP_DEPTH_CLAMP_RANGE)) {
      struct pan_ptr vp = panvk_cmd_alloc_desc(cmdbuf, VIEWPORT);
      if (!vp.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      panvk_emit_viewport(cmdbuf, vp.cpu);
      cmdbuf->state.gfx.vpd = vp.gpu;
   }

   draw->viewport = cmdbuf->state.gfx.vpd;
   return VK_SUCCESS;
}

static void
panvk_emit_vertex_dcd(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_draw_data *draw,
                      struct mali_draw_packed *dcd)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;

   pan_pack(dcd, DRAW, cfg) {
      cfg.state = panvk_priv_mem_dev_addr(vs->rsd);
      cfg.attributes = draw->vs.attributes;
      cfg.attribute_buffers = draw->vs.attribute_bufs;
      cfg.varyings = draw->vs.varyings;
      cfg.varying_buffers = draw->varying_bufs;
      cfg.thread_storage = draw->tls;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
      }

      cfg.uniform_buffers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.vs.push_uniforms;
      cfg.textures = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = vs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];
   }
}

static VkResult
panvk_draw_prepare_vertex_job(struct panvk_cmd_buffer *cmdbuf,
                              struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.vertex = ptr;

   memcpy(pan_section_ptr(ptr.cpu, COMPUTE_JOB, INVOCATION), &draw->invocation,
          pan_size(INVOCATION));

   pan_section_pack(ptr.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = 5;
   }

   panvk_emit_vertex_dcd(cmdbuf, draw,
                         pan_section_ptr(ptr.cpu, COMPUTE_JOB, DRAW));
   return VK_SUCCESS;
}

static void
panvk_emit_tiler_primitive(struct panvk_cmd_buffer *cmdbuf,
                           const struct panvk_draw_data *draw,
                           struct mali_primitive_packed *prim)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_input_assembly_state *ia = &dyns->ia;
   const struct vk_rasterization_state *rs = &dyns->rs;
   bool writes_point_size =
      vs->info.vs.writes_point_size &&
      ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool secondary_shader = vs->info.vs.secondary_enable && fs != NULL;
   assert(!(vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID));
   bool fs_reads_primitive_id = fs ? fs->info.fs.reads_primitive_id : false;

   pan_pack(prim, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->info.prim);
      if (writes_point_size)
         cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP16;
      cfg.primitive_index_enable = fs_reads_primitive_id;
      cfg.primitive_index_writeback = fs_reads_primitive_id;

      cfg.first_provoking_vertex =
         cmdbuf->state.gfx.render.first_provoking_vertex != U_TRISTATE_NO;

      if (draw->info.index.restart_enable)
         cfg.primitive_restart = MALI_PRIMITIVE_RESTART_IMPLICIT;
      cfg.job_task_split = 6;

      if (draw->info.index.index_size) {
         switch (draw->info.index.index_size) {
         case 4:
            cfg.index_type = MALI_INDEX_TYPE_UINT32;
            break;
         case 2:
            cfg.index_type = MALI_INDEX_TYPE_UINT16;
            break;
         case 1:
            cfg.index_type = MALI_INDEX_TYPE_UINT8;
            break;
         default:
            UNREACHABLE("Invalid index size");
         }
      }

      /* In case of indirect draw, the descriptor will be patched at runtime */
      cfg.index_count = is_indirect_draw(draw) ? 1 : draw->info.vertex.count;

      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);

      cfg.secondary_shader = secondary_shader;
   }
}

static void
panvk_emit_tiler_primitive_size(struct panvk_cmd_buffer *cmdbuf,
                                const struct panvk_draw_data *draw,
                                struct mali_primitive_size_packed *primsz)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const bool writes_point_size =
      vs->info.vs.writes_point_size && draw->info.prim == MESA_PRIM_POINTS;

   pan_pack(primsz, PRIMITIVE_SIZE, cfg) {
      if (writes_point_size) {
         cfg.size_array = draw->psiz;
      } else {
         cfg.fixed_sized = draw->line_width;
      }
   }
}

static void
panvk_emit_tiler_dcd(struct panvk_cmd_buffer *cmdbuf,
                     const struct panvk_draw_data *draw,
                     struct mali_draw_packed *dcd)
{
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;

   enum mesa_prim reduced_prim = u_reduced_prim(draw->info.prim);
   const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;

   pan_pack(dcd, DRAW, cfg) {
      cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
      cfg.cull_front_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.cull_back_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      cfg.position = draw->position;
      cfg.state = draw->fs.rsd;
      cfg.attributes = fs_desc_state->img_attrib_table;
      cfg.attribute_buffers =
         fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_IMG];
      cfg.viewport = draw->viewport;
      cfg.varyings = draw->fs.varyings;
      cfg.varying_buffers = cfg.varyings ? draw->varying_bufs : 0;
      cfg.thread_storage = draw->tls;

      /* For all primitives but lines DRAW.flat_shading_vertex must
       * be set to 0 and the provoking vertex is selected with the
       * PRIMITIVE.first_provoking_vertex field.
       */
      if (reduced_prim == MESA_PRIM_LINES)
         cfg.flat_shading_vertex = true;

      /* In case of indirect draw, the descriptor will be patched at runtime */
      if (!is_indirect_draw(draw)) {
         cfg.offset_start = draw->info.vertex.raw_offset;
         cfg.instance_size =
            draw->info.instance.count > 1 ? draw->padded_vertex_count : 1;
         uint32_t primitives_per_instance =
            DIV_ROUND_UP(draw->padded_vertex_count,
                         mesa_vertices_per_prim(draw->info.prim));
         /* instance_primitive_size has the same restrictions as
          * padded_vertex_count, so we can use pan_padded_vertex_count here. */
         cfg.instance_primitive_size =
            pan_padded_vertex_count(primitives_per_instance);
      }

      cfg.uniform_buffers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_UBO];
      cfg.push_uniforms = cmdbuf->state.gfx.fs.push_uniforms;
      cfg.textures = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_TEXTURE];
      cfg.samplers = fs_desc_state->tables[PANVK_BIFROST_DESC_TABLE_SAMPLER];

      cfg.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;
#if PAN_ARCH == 9
      cfg.scissor_to_bounding_box = true;
#endif
   }
}

static VkResult
panvk_draw_prepare_tiler_job(struct panvk_cmd_buffer *cmdbuf,
                             struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, TILER_JOB);
   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.tiler = ptr;

   memcpy(pan_section_ptr(ptr.cpu, TILER_JOB, INVOCATION), &draw->invocation,
          pan_size(INVOCATION));

   panvk_emit_tiler_primitive(cmdbuf, draw,
                              pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, TILER_JOB, PRIMITIVE_SIZE));

   panvk_emit_tiler_dcd(cmdbuf, draw,
                        pan_section_ptr(ptr.cpu, TILER_JOB, DRAW));

   pan_section_pack(ptr.cpu, TILER_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, TILER_JOB, PADDING, padding)
      ;

   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_idvs_job(struct panvk_cmd_buffer *cmdbuf,
                            struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, INDEXED_VERTEX_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->jobs.idvs = ptr;

   memcpy(pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, INVOCATION),
          &draw->invocation, pan_size(INVOCATION));

   panvk_emit_tiler_primitive(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE));

   panvk_emit_tiler_primitive_size(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, PRIMITIVE_SIZE));

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, TILER, cfg) {
      cfg.address = PAN_ARCH >= 9 ? draw->tiler_ctx->valhall.desc
                                  : draw->tiler_ctx->bifrost.desc;
   }

   pan_section_pack(ptr.cpu, INDEXED_VERTEX_JOB, PADDING, _) {
   }

   panvk_emit_tiler_dcd(
      cmdbuf, draw,
      pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, FRAGMENT_DRAW));

   panvk_emit_vertex_dcd(
      cmdbuf, draw, pan_section_ptr(ptr.cpu, INDEXED_VERTEX_JOB, VERTEX_DRAW));
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_vs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_state *vs_desc_state =
      &cmdbuf->state.gfx.vs.desc;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;
   unsigned num_vbs = util_last_bit(vi->bindings_valid);
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, vs_desc_info, &cmdbuf->state.gfx.desc_state, vs_desc_state,
      num_vbs * pan_size(ATTRIBUTE_BUFFER) * 2, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.vertex_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_draw_prepare_fs_copy_desc_job(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_draw_data *draw)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr ptr;

   VkResult result = panvk_per_arch(meta_get_copy_desc_job)(
      cmdbuf, fs_desc_info, &cmdbuf->state.gfx.desc_state,
      fs_desc_state, 0, &ptr);
   if (result != VK_SUCCESS)
      return result;

   if (ptr.cpu) {
      util_dynarray_append(&batch->jobs, ptr.cpu);
   }

   draw->jobs.frag_copy_desc = ptr;
   return VK_SUCCESS;
}

static VkResult
panvk_cmd_prepare_draw_link_shaders(struct panvk_cmd_buffer *cmd)
{
   struct panvk_cmd_graphics_state *gfx = &cmd->state.gfx;

   if (!gfx_state_dirty(cmd, VS) && !gfx_state_dirty(cmd, FS))
      return VK_SUCCESS;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmd->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmd));

   VkResult result =
      panvk_per_arch(link_shaders)(&cmd->desc_pool, vs, fs, &gfx->link);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return result;
   }

   return VK_SUCCESS;
}

static VkResult
prepare_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   VkResult result;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   /* There are only 16 bits in the descriptor for the job ID. Each job has a
    * pilot shader dealing with descriptor copies, and we need one
    * <vertex,tiler> pair per draw.
    */
   if (batch->vtc_jc.job_index + (4 * cmdbuf->state.gfx.render.layer_count) >=
       UINT16_MAX) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }

   if (fs_user_dirty(cmdbuf)) {
      result = panvk_cmd_prepare_draw_link_shaders(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   if (cmdbuf->state.gfx.vk_meta) {
      /* vk_meta doesn't care about the provoking vertex mode, we should use
       * the same mode that the application uses. */
      set_provoking_vertex_mode(cmdbuf, U_TRISTATE_UNSET);
   } else {
      enum u_tristate first_provoking_vertex = u_tristate_make(
         cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
         VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
      set_provoking_vertex_mode(cmdbuf, first_provoking_vertex);
   }
   /* Changing the provoking vertex mode may have split the batch. */
   batch = cmdbuf->cur_batch;

   if (!rs->rasterizer_discard_enable) {
      ASSERTED const struct pan_fb_layout *fb =
         &cmdbuf->state.gfx.render.fb.layout;
      uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
      uint32_t rasterization_samples =
         cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

      /* If there's no attachment, and the FB descriptor hasn't been allocated
       * yet, we patch nr_samples to match rasterization_samples, otherwise, we
       * make sure those two numbers match. */
      if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments) {
         assert(rasterization_samples > 0);
         *nr_samples = rasterization_samples;
      } else {
         assert(rasterization_samples == *nr_samples);
      }

      /* In case we already emitted tiler/framebuffer descriptors, we ensure
       * that the sample count didn't change
       * XXX: This currently can happen in case we resume a render pass with no
       * attachements and without any draw as the FBD is emitted when suspending.
       */
      assert(fb->sample_count == 0 ||
             fb->sample_count == cmdbuf->state.gfx.render.fb.nr_samples);

      result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   const struct panvk_shader_desc_info *fs_desc_info =
      fs ? &cmdbuf->state.gfx.fs.shader->desc_info : NULL;

   uint32_t used_set_mask =
      vs_desc_info->used_set_mask | (fs ? fs_desc_info->used_set_mask : 0);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS) ||
       gfx_state_dirty(cmdbuf, FS)) {
      result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                                      used_set_mask);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS)) {
      result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
         cmdbuf, desc_state, vs_desc_info, false, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;

      result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
         cmdbuf, desc_state, vs_desc_info, vs_desc_state);
      if (result != VK_SUCCESS)
         return result;
   }

   /* This allocates and initializes the image table, which we need before we
    * can copy descriptors.
    */
   panvk_draw_prepare_attributes(cmdbuf, draw);

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS)) {
      result = panvk_draw_prepare_vs_copy_desc_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   }

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, FS)) {
      if (fs == NULL) {
         /* No need to setup the FS desc tables if the FS is not executed. */
         memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      } else {
         result = panvk_per_arch(cmd_prepare_shader_desc_tables)(
            cmdbuf, desc_state, fs_desc_info, true, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_per_arch(cmd_prepare_dyn_ssbos)(
            cmdbuf, desc_state, fs_desc_info, fs_desc_state);
         if (result != VK_SUCCESS)
            return result;

         result = panvk_draw_prepare_fs_copy_desc_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   draw->tls = batch->tls.gpu;
   draw->fb = batch->fb.desc.gpu;

   result = panvk_draw_prepare_fs_rsd(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, fs);

   /* Viewport emission requires up-to-date {scale,offset}.z for min/max Z,
    * so we need to call it after calling cmd_prepare_draw_sysvals(), but
    * viewports are the same for all layers, so we only emit when layer_id=0.
    */
   result = panvk_draw_prepare_viewport(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   return VK_SUCCESS;
}

static VkResult
prepare_draw_layer(struct panvk_cmd_buffer *cmdbuf,
                   struct panvk_draw_data *draw, uint32_t layer)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   VkResult result;

   result = panvk_draw_prepare_varyings(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   draw->info.layer_id = layer;
   if (draw->info.layer_id > 0) {
      cmdbuf->state.gfx.sysvals.layer_id = draw->info.layer_id;
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);
   }

   struct pan_ptr vs_push_uniforms;
   result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
      cmdbuf, vs, &vs_push_uniforms, 1);
   if (result != VK_SUCCESS)
      return result;
   cmdbuf->state.gfx.vs.push_uniforms = vs_push_uniforms.gpu;

   if (fs) {
      struct pan_ptr fs_push_uniforms;
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, fs, &fs_push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.gfx.fs.push_uniforms = fs_push_uniforms.gpu;
   }

   result = panvk_draw_prepare_tiler_context(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   if (vs->info.vs.idvs) {
      result = panvk_draw_prepare_idvs_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;
   } else {
      result = panvk_draw_prepare_vertex_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return result;

      bool needs_tiling =
         !cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable ||
         cmdbuf->state.gfx.occlusion_query.mode !=
            MALI_OCCLUSION_MODE_DISABLED;

      if (needs_tiling) {
         result = panvk_draw_prepare_tiler_job(cmdbuf, draw);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   return VK_SUCCESS;
}

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   pan_pack_work_groups_compute(&draw->invocation, 1, draw->vertex_range,
                                draw->info.instance.count, 1, 1, 1, true,
                                false);

   struct panvk_batch *batch = cmdbuf->cur_batch;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;
      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, 0, copy_desc_job_id, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, false,
                           0, copy_desc_job_id, &draw->jobs.vertex, false);

         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, false,
                           vjob_id, 0, &draw->jobs.tiler, false);
         }
      }
   }

   clear_dirty_after_draw(cmdbuf);
}

static void
panvk_cmd_draw_indirect(struct panvk_cmd_buffer *cmdbuf,
                        struct panvk_draw_data *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->rsd))
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct vk_vertex_input_state *vi =
      cmdbuf->vk.dynamic_graphics_state.vi;

   unsigned init_heap_job_id;
   result = panvk_per_arch(cmd_init_poly_heap)(cmdbuf, &init_heap_job_id);
   if (result != VK_SUCCESS)
      return;

   unsigned copy_desc_job_id =
      draw->jobs.vertex_copy_desc.gpu
         ? pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false,
                          0, 0, &draw->jobs.vertex_copy_desc, false)
         : 0;

   if (draw->jobs.frag_copy_desc.gpu) {
      /* We don't need to add frag_copy_desc as a dependency because the
       * tiler job doesn't execute the fragment shader, the fragment job
       * will, and the tiler/fragment synchronization happens at the batch
       * level. */
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0,
                     &draw->jobs.frag_copy_desc, false);
   }

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   uint64_t index_min_max_res_ptr = 0;
   uint32_t job_before_indirect_helper = copy_desc_job_id;
   if (draw->info.index.index_size) {
      index_min_max_res_ptr =
         panvk_cmd_alloc_dev_mem(
            cmdbuf, desc,
            sizeof(struct libpan_draw_helper_index_min_max_result), 8)
            .gpu;
      const struct panlib_draw_index_minmax_search_helper_args args = {
         .index_buffer_ptr = draw->info.index.buffer_dev_addr,
         .cmd = draw->info.indirect.buffer_dev_addr,
         .min_ptr =
            index_min_max_res_ptr +
            offsetof(struct libpan_draw_helper_index_min_max_result, min),
         .max_ptr =
            index_min_max_res_ptr +
            offsetof(struct libpan_draw_helper_index_min_max_result, max),
      };

      struct libpan_draw_helper_index_min_max_result val = {
         .min = ((uint64_t)1 << (draw->info.index.index_size * 8)) - 1,
         .max = 0,
      };
      uint64_t *raw_val = (uint64_t *)&val;

      struct pan_ptr write_job =
         pan_pool_alloc_desc(&cmdbuf->desc_pool.base, WRITE_VALUE_JOB);

      pan_section_pack(write_job.cpu, WRITE_VALUE_JOB, PAYLOAD, payload) {
         payload.type = MALI_WRITE_VALUE_TYPE_IMMEDIATE_64;
         payload.address = index_min_max_res_ptr;
         payload.immediate_value = *raw_val;
      };

      unsigned write_job_id =
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_WRITE_VALUE, false, false,
                        0, copy_desc_job_id, &write_job, false);
      util_dynarray_append(&batch->jobs, write_job.cpu);

      const uint32_t index_count =
         draw->info.index.buffer_size / draw->info.index.index_size;
      uint32_t wg_count = DIV_ROUND_UP(index_count, 65536);
      assert(wg_count <= 65536);

      panlib_draw_index_minmax_search_helper_struct(
         &precomp_ctx, panlib_1d_with_jm_deps(wg_count, 0, write_job_id),
         PANLIB_BARRIER_NONE, args, util_logbase2(draw->info.index.index_size),
         draw->info.index.restart_enable);
      job_before_indirect_helper = batch->vtc_jc.job_index;
   }

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;

      /* Force a new push uniform block to be allocated */
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);

      result = prepare_draw_layer(cmdbuf, draw, layer);
      if (result != VK_SUCCESS)
         return;

      assert(draw->info.indirect.buffer_dev_addr != 0 ||
             draw->info.index.index_size);

      uint32_t attrib_bufs_valid = vi->bindings_valid;
      uint32_t attribs_valid = vi->attributes_valid;
      uint64_t first_vertex_sysval = 0x8ull << 60;
      uint64_t first_instance_sysval = 0x8ull << 60;
      uint64_t raw_vertex_offset_sysval = 0x8ull << 60;
      if (shader_uses_sysval(vs, graphics, vs.first_vertex)) {
         first_vertex_sysval = cmdbuf->state.gfx.vs.push_uniforms +
                               shader_remapped_sysval_offset(
                                  vs, sysval_offset(graphics, vs.first_vertex));
      }

      if (shader_uses_sysval(vs, graphics, vs.base_instance)) {
         first_instance_sysval =
            cmdbuf->state.gfx.vs.push_uniforms +
            shader_remapped_sysval_offset(
               vs, sysval_offset(graphics, vs.base_instance));
      }

      if (shader_uses_sysval(vs, graphics, vs.raw_vertex_offset)) {
         raw_vertex_offset_sysval =
            cmdbuf->state.gfx.vs.push_uniforms +
            shader_remapped_sysval_offset(
               vs, sysval_offset(graphics, vs.raw_vertex_offset));
      }

      enum panlib_barrier indirect_barrier =
         PANLIB_BARRIER_JM_SUPPRESS_PREFETCH;
      struct panlib_precomp_grid indirect_grid = panlib_1d_with_jm_deps(
         1, init_heap_job_id, job_before_indirect_helper);

      if (draw->info.indirect.buffer_dev_addr != 0 &&
          draw->info.index.index_size) {
         const struct panlib_draw_indexed_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .heap = panvk_priv_mem_dev_addr(dev->poly_heap.state),
            .index_buffer_ptr = draw->info.index.buffer_dev_addr,
            .index_min_max_res = index_min_max_res_ptr,
            .index_size = draw->info.index.index_size,
            .primitive_vertex_count =
               mesa_vertices_per_prim(draw->info.prim),
            .varying_bufs_descs = draw->varying_bufs,
            .attrib_bufs_descs = draw->vs.attribute_bufs,
            .attrib_bufs_infos = draw->indirect_info.attrib_bufs,
            .attrib_bufs_valid = attrib_bufs_valid,
            .attribs_valid = attribs_valid,
            .attribs_descs = draw->vs.attributes,
            .attribs_infos = draw->indirect_info.attribs,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .raw_vertex_offset_sysval = raw_vertex_offset_sysval,
            .idvs_job = vs->info.vs.idvs ? draw->jobs.idvs.gpu : 0,
            .vertex_job = draw->jobs.vertex.gpu,
            .tiler_job = draw->jobs.tiler.gpu,
         };
         panlib_draw_indexed_indirect_helper_struct(&precomp_ctx, indirect_grid,
                                                    indirect_barrier, args);
      } else if (draw->info.indirect.buffer_dev_addr != 0) {
         const struct panlib_draw_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .primitive_vertex_count =
               mesa_vertices_per_prim(draw->info.prim),
            .heap = panvk_priv_mem_dev_addr(dev->poly_heap.state),
            .varying_bufs_descs = draw->varying_bufs,
            .attrib_bufs_descs = draw->vs.attribute_bufs,
            .attrib_bufs_infos = draw->indirect_info.attrib_bufs,
            .attrib_bufs_valid = attrib_bufs_valid,
            .attribs_valid = attribs_valid,
            .attribs_descs = draw->vs.attributes,
            .attribs_infos = draw->indirect_info.attribs,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .raw_vertex_offset_sysval = raw_vertex_offset_sysval,
            .idvs_job = vs->info.vs.idvs ? draw->jobs.idvs.gpu : 0,
            .vertex_job = draw->jobs.vertex.gpu,
            .tiler_job = draw->jobs.tiler.gpu,
         };
         panlib_draw_indirect_helper_struct(&precomp_ctx, indirect_grid,
                                            indirect_barrier, args);
      } else {
         assert(false && "Invalid indirect draw");
      }

      /* Grab the index of the indirect helper job */
      uint32_t prev_job = batch->vtc_jc.job_index;

      if (vs->info.vs.idvs) {
         pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_INDEXED_VERTEX, false,
                        false, 0, prev_job, &draw->jobs.idvs, false);
      } else {
         unsigned vjob_id =
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_VERTEX, false, true, 0,
                           prev_job, &draw->jobs.vertex, false);

         if (draw->jobs.tiler.gpu != 0) {
            pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_TILER, false, false,
                           vjob_id, 0, &draw->jobs.tiler, false);
         }
      }
   }

   /*
    * We split every ~1024 indirect draw.
    * This is here for multiple reasons:
    * - The heap offset need to be reset at some point to avoid going outside of
    * bounds.
    * - It is possible to always end up with timeouts for batches with 4k draws
    * (see "dEQP-VK.api.command_buffers.many_indirect_draws_on_secondary") At
    * the same time, because of how TLS works on Mali, we should not split too
    * much as this will cause the TLS budget to go crazy.
    */
   if (batch->vtc_jc.job_index > (5 * 1024)) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
      cmdbuf->state.uses_poly_heap = false;
   }

   clear_dirty_after_draw(cmdbuf);
}

static unsigned
padded_vertex_count(struct panvk_cmd_buffer *cmdbuf, uint32_t vertex_count,
                    uint32_t instance_count)
{
   if (instance_count == 1)
      return vertex_count;

   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   bool idvs = vs->info.vs.idvs;

   /* Index-Driven Vertex Shading requires different instances to
    * have different cache lines for position results. Each vertex
    * position is 16 bytes and the Mali cache line is 64 bytes, so
    * the instance count must be aligned to 4 vertices.
    */
   if (idvs)
      vertex_count = ALIGN_POT(vertex_count, 4);

   return pan_padded_vertex_count(vertex_count);
}

#else /* PAN_ARCH >= 9 */

/*
 * Valhall (v9) draw path for the JM backend.
 *
 * v9 uses Valhall descriptors (resource tables, shader program descriptors,
 * FAU) like the CSF backend, but jobs are still submitted as job chains like
 * on Bifrost. Each draw (and each layer) is a single MALLOC_VERTEX_JOB: an
 * IDVS job where the hardware allocates the varyings. The descriptor setup
 * below mirrors csf/panvk_vX_cmd_draw.c, the job layout mirrors
 * jm_emit_malloc_vertex_job() in the Gallium driver.
 */

static void
prepare_vi(struct panvk_cmd_buffer *cmdbuf)
{
   if (!dyn_gfx_state_dirty(cmdbuf, VI) &&
       !dyn_gfx_state_dirty(cmdbuf, VI_BINDINGS_VALID))
      return;

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   cmdbuf->state.gfx.vi.attribs_changing_on_base_instance = 0;
   u_foreach_bit(i, vi->attributes_valid) {
      const struct vk_vertex_binding_state *binding =
         &vi->bindings[vi->attributes[i].binding];
      const uint32_t stride =
         dyns->vi_binding_strides[vi->attributes[i].binding];

      if (binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE && stride != 0) {
         cmdbuf->state.gfx.vi.attribs_changing_on_base_instance |=
            BITFIELD_BIT(i);
      }
   }
}

static void
emit_vs_attrib(struct panvk_cmd_buffer *cmdbuf, uint32_t attrib_idx,
               uint32_t vb_desc_offset, struct mali_attribute_packed *desc)
{
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   const struct vk_vertex_attribute_state *attrib_info =
      &vi->attributes[attrib_idx];
   const struct vk_vertex_binding_state *buf_info =
      &vi->bindings[attrib_info->binding];
   const uint32_t stride = dyns->vi_binding_strides[attrib_info->binding];
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = vb_desc_offset + attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.offset = attrib_info->offset;

      if (per_instance)
         cfg.offset += cmdbuf->state.gfx.vi.base_instance * stride;

      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;
      cfg.table = 0;
      cfg.buffer_index = buf_idx;
      cfg.stride = stride;
      if (!per_instance) {
         /* Per-vertex */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset_enable = true;
      } else if (buf_info->divisor == 1) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      } else if (buf_info->divisor == 0) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         /* HW doesn't support a zero divisor, but we can achieve the same by
          * not using a divisor and setting the stride to zero */
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.stride = 0;
      } else if (util_is_power_of_two_or_zero(buf_info->divisor)) {
         /* Per-instance, POT divisor */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_r = __builtin_ctz(buf_info->divisor);
      } else {
         /* Per-instance, NPOT divisor */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_d = pan_compute_npot_divisor(
            buf_info->divisor, &cfg.divisor_r, &cfg.divisor_e);
      }
   }
}

static VkResult
prepare_vs_driver_set(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   uint32_t vb_count = 0;
   u_foreach_bit(i, vi->attributes_valid)
      vb_count = MAX2(vi->attributes[i].binding + 1, vb_count);

   uint32_t vb_offset = vs_desc_info->dyn_bufs.count + MAX_VS_ATTRIBS + 1;
   uint32_t desc_count = vb_offset + vb_count;

   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (!driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   for (uint32_t i = 0; i < MAX_VS_ATTRIBS; i++) {
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         emit_vs_attrib(cmdbuf, i, vb_offset,
                        (struct mali_attribute_packed *)(&descs[i]));
      } else {
         /* Write a specialized AttributeDescriptor and rely on OOB behavior */
         pan_cast_and_pack(&descs[i], ATTRIBUTE, cfg) {
            cfg.table = 17; /* Invalid table, ensuring OOB access */
            cfg.format = (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
         }
      }
   }

   /* Dummy sampler always comes right after the vertex attribs. */
   pan_cast_and_pack(&descs[MAX_VS_ATTRIBS], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }

   panvk_per_arch(cmd_fill_dyn_bufs)(
      desc_state, vs_desc_info,
      (struct mali_buffer_packed *)(&descs[MAX_VS_ATTRIBS + 1]));

   for (uint32_t i = 0; i < vb_count; i++) {
      const struct panvk_attrib_buf *vb = &cmdbuf->state.gfx.vb.bufs[i];
      const bool nulldesc = (vb->address == 0 && vb->size == 0);

      if ((vi->bindings_valid & BITFIELD_BIT(i)) && !nulldesc) {
         pan_cast_and_pack(&descs[vb_offset + i], BUFFER, cfg) {
            cfg.address = vb->address;
            cfg.size = vb->size;
         }
      } else {
         /* Write a NullDescriptor and rely on OOB behavior */
         pan_cast_and_pack(&descs[vb_offset + i], NULL_DESCRIPTOR, cfg)
            ;
      }
   }

   vs_desc_state->driver_set.dev_addr = driver_set.gpu;
   vs_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   gfx_state_set_dirty(cmdbuf, DESC_STATE);
   return VK_SUCCESS;
}

/* Only valid to call after prepare_vi() */
static bool
vs_desc_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   /* If any of the attributes change on base instance and the base instance
    * has changed, we need to re-emit regardless of what API state is dirty.
    */
   if (cmdbuf->state.gfx.vi.attribs_changing_on_base_instance &&
       gfx_state_dirty(cmdbuf, BASE_INSTANCE))
      return true;

   return dyn_gfx_state_dirty(cmdbuf, VI) ||
          dyn_gfx_state_dirty(cmdbuf, VI_BINDINGS_VALID) ||
          dyn_gfx_state_dirty(cmdbuf, VI_BINDING_STRIDES) ||
          gfx_state_dirty(cmdbuf, VB) || gfx_state_dirty(cmdbuf, VS) ||
          gfx_state_dirty(cmdbuf, DESC_STATE);
}

static VkResult
prepare_vs_desc(struct panvk_cmd_buffer *cmdbuf)
{
   prepare_vi(cmdbuf);

   if (!vs_desc_dirty(cmdbuf))
      return VK_SUCCESS;

   VkResult result = prepare_vs_driver_set(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   return panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &cmdbuf->state.gfx.desc_state,
      &cmdbuf->state.gfx.vs.shader->desc_info, &cmdbuf->state.gfx.vs.desc,
      1);
}

static void
emit_varying_descs(const struct panvk_cmd_buffer *cmdbuf,
                   struct mali_attribute_packed *descs)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   const struct pan_varying_layout *vs_layout = &vs->info.varyings.formats;
   const struct pan_varying_layout *fs_format = &fs->info.varyings.formats;
   pan_varying_layout_require_layout(vs_layout);
   pan_varying_layout_require_format(fs_format);

   for (uint32_t i = 0; i < fs_format->count; i++) {
      const struct pan_varying_slot *fs_slot =
         pan_varying_layout_slot_at(fs_format, i);

      /* Skip empty slots and special varyings. */
      if (!fs_slot || fs_slot->section != PAN_VARYING_SECTION_GENERIC)
         continue;

      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;

      const struct pan_varying_slot *vs_slot =
         pan_varying_layout_find_slot(vs_layout, fs_slot->location);
      if (vs_slot) {
         nir_alu_type base_type = nir_alu_type_get_base_type(fs_slot->alu_type);
         nir_alu_type bit_size = nir_alu_type_get_type_size(vs_slot->alu_type);

         offset = vs_slot->offset;
         format = pan_varying_format(base_type | bit_size, vs_slot->ncomps);
      }

      /* Varyings are read from the hardware-allocated vertex packets
       * (table 61, buffer 0 before v12). */
      pan_pack(&descs[i], ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         cfg.buffer_index = 0;
         cfg.attribute_stride = vs_layout->generic_size_B;
         cfg.packet_stride = vs_layout->generic_size_B + 16;
      }
   }
}

static VkResult
prepare_fs_driver_set(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   /* If the shader is using LD_VAR_BUF[_IMM], we do not have to set up
    * Attribute Descriptors for varying loads. */
   const uint32_t desc_count = fs_desc_info->fs_varying_attr_desc_count +
                               fs_desc_info->dyn_bufs.count + 1;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (desc_count && !driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (fs_desc_info->fs_varying_attr_desc_count > 0)
      emit_varying_descs(cmdbuf, (struct mali_attribute_packed *)(&descs[0]));

   /* Dummy sampler always comes right after the varyings. */
   const uint32_t sampler_idx = fs_desc_info->fs_varying_attr_desc_count;
   pan_cast_and_pack(&descs[sampler_idx], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }

   panvk_per_arch(cmd_fill_dyn_bufs)(
      desc_state, fs_desc_info,
      (struct mali_buffer_packed *)(&descs[sampler_idx + 1]));

   fs_desc_state->driver_set.dev_addr = driver_set.gpu;
   fs_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   gfx_state_set_dirty(cmdbuf, DESC_STATE);
   return VK_SUCCESS;
}

static bool
fs_desc_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   return fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, VS) ||
          gfx_state_dirty(cmdbuf, DESC_STATE);
}

static VkResult
prepare_fs_desc(struct panvk_cmd_buffer *cmdbuf)
{
   if (!fs_desc_dirty(cmdbuf))
      return VK_SUCCESS;

   const struct panvk_shader *fs = get_fs(cmdbuf);
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;

   if (!fs) {
      memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      return VK_SUCCESS;
   }

   VkResult result = prepare_fs_driver_set(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   return panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, &cmdbuf->state.gfx.desc_state, &fs->desc_info, fs_desc_state,
      1);
}

static VkResult
prepare_descs(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct panvk_shader *fs = get_fs(cmdbuf);
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   VkResult result;

   if (gfx_state_dirty(cmdbuf, DESC_STATE) || gfx_state_dirty(cmdbuf, VS) ||
       fs_user_dirty(cmdbuf)) {
      uint32_t used_set_mask = vs->desc_info.used_set_mask;
      used_set_mask |= fs ? fs->desc_info.used_set_mask : 0;

      result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                                      used_set_mask);
      if (result != VK_SUCCESS)
         return result;
   }

   result = prepare_vs_desc(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   return prepare_fs_desc(cmdbuf);
}

static bool
zsd_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   return !cmdbuf->state.gfx.dcd.zsd ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_COMPARE_MASK) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_REFERENCE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_FACTORS) ||
          dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_COVERAGE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, INPUT_ATTACHMENT_MAP) ||
          fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, OQ) ||
          /* has_{depth,stencil}_att() depend on the render pass */
          gfx_state_dirty(cmdbuf, RENDER_STATE);
}

static VkResult
prepare_zsd(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   if (!zsd_dirty(cmdbuf)) {
      draw->zsd = cmdbuf->state.gfx.dcd.zsd;
      return VK_SUCCESS;
   }

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;
   const struct vk_depth_stencil_state *ds = &dyns->ds;
   bool test_s = has_stencil_att(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att(cmdbuf) && ds->depth.test_enable;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   struct pan_ptr zsd = panvk_cmd_alloc_desc(cmdbuf, DEPTH_STENCIL);
   if (!zsd.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = test_s;
      if (test_s) {
         cfg.front_compare_function =
            translate_compare_func(ds->stencil.front.op.compare);
         cfg.front_stencil_fail =
            translate_stencil_op(ds->stencil.front.op.fail);
         cfg.front_depth_fail =
            translate_stencil_op(ds->stencil.front.op.depth_fail);
         cfg.front_depth_pass = translate_stencil_op(ds->stencil.front.op.pass);
         cfg.back_compare_function =
            translate_compare_func(ds->stencil.back.op.compare);
         cfg.back_stencil_fail = translate_stencil_op(ds->stencil.back.op.fail);
         cfg.back_depth_fail =
            translate_stencil_op(ds->stencil.back.op.depth_fail);
         cfg.back_depth_pass = translate_stencil_op(ds->stencil.back.op.pass);
      }

      cfg.stencil_from_shader = fs ? fs->info.fs.writes_stencil : 0;
      cfg.front_write_mask = ds->stencil.front.write_mask;
      cfg.back_write_mask = ds->stencil.back.write_mask;
      cfg.front_value_mask = ds->stencil.front.compare_mask;
      cfg.back_value_mask = ds->stencil.back.compare_mask;
      cfg.front_reference_value = ds->stencil.front.reference;
      cfg.back_reference_value = ds->stencil.back.reference;

      cfg.depth_cull_enable = vk_rasterization_state_depth_clip_enable(rs);
      if (rs->depth_clamp_enable)
         cfg.depth_clamp_mode = MALI_DEPTH_CLAMP_MODE_BOUNDS;

      if (fs)
         cfg.depth_source = pan_depth_source(&fs->info);

      cfg.depth_write_enable = test_z && ds->depth.write_enable;
      cfg.depth_bias_enable = rs->depth_bias.enable;
      cfg.depth_function = test_z ? translate_compare_func(ds->depth.compare_op)
                                  : MALI_FUNC_ALWAYS;
      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;
   }

   cmdbuf->state.gfx.dcd.zsd = zsd.gpu;
   draw->zsd = zsd.gpu;
   return VK_SUCCESS;
}

static bool
blend_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   return !cmdbuf->state.gfx.dcd.blend_valid ||
          dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP) ||
          dyn_gfx_state_dirty(cmdbuf, CB_ATTACHMENT_COUNT) ||
          dyn_gfx_state_dirty(cmdbuf, CB_COLOR_WRITE_ENABLES) ||
          dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
          dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
          dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
          dyn_gfx_state_dirty(cmdbuf, CB_BLEND_CONSTANTS) ||
          dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP) ||
          fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, RENDER_STATE);
}

static VkResult
prepare_blend(struct panvk_cmd_buffer *cmdbuf,
              const struct panvk_shader_variant *fs,
              struct panvk_draw_data *draw)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;

   if (!blend_dirty(cmdbuf)) {
      draw->blend = gfx->dcd.blend;
      draw->blend_count = gfx->dcd.blend_count;
      return VK_SUCCESS;
   }

   uint32_t bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
   struct pan_ptr ptr = panvk_cmd_alloc_desc_array(cmdbuf, bd_count, BLEND);
   struct mali_blend_packed *bds = ptr.cpu;

   if (bd_count && !ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (fs) {
      VkResult result = panvk_per_arch(blend_emit_descs)(cmdbuf, bds);
      if (result != VK_SUCCESS)
         return result;
   } else {
      for (unsigned i = 0; i < bd_count; i++) {
         pan_pack(&bds[i], BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
         }
      }
   }

   gfx->dcd.blend_valid = true;
   gfx->dcd.blend = ptr.gpu;
   gfx->dcd.blend_count = bd_count;
   draw->blend = ptr.gpu;
   draw->blend_count = bd_count;
   return VK_SUCCESS;
}

/* Same logic as build_dcd_flags() in the CSF backend, minus DCD flags 2,
 * which doesn't exist on v9. */
static bool
dcd_flags_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   /* Union of the CSF DCD0/DCD1 dirty conditions. */
   return !cmdbuf->state.gfx.dcd.flags_valid ||
          dyn_gfx_state_dirty(cmdbuf, RS_RASTERIZER_DISCARD_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_CULL_MODE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_LINE_MODE) ||
          dyn_gfx_state_dirty(cmdbuf, RS_FRONT_FACE) ||
          dyn_gfx_state_dirty(cmdbuf, MS_RASTERIZATION_SAMPLES) ||
          dyn_gfx_state_dirty(cmdbuf, MS_SAMPLE_MASK) ||
          dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_COVERAGE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
          dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
          dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
          dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
          dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
          dyn_gfx_state_dirty(cmdbuf, IA_PRIMITIVE_TOPOLOGY) ||
          dyn_gfx_state_dirty(cmdbuf, INPUT_ATTACHMENT_MAP) ||
          dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP) ||
          fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, RENDER_STATE) ||
          gfx_state_dirty(cmdbuf, OQ);
}

static void
prepare_dcd_flags(struct panvk_cmd_buffer *cmdbuf,
                  const struct panvk_shader_variant *fs,
                  struct panvk_draw_data *draw)
{
   struct panvk_cmd_graphics_state *gfx = &cmdbuf->state.gfx;

   if (!dcd_flags_dirty(cmdbuf)) {
      draw->dcd_flags_0 = gfx->dcd.flags_0;
      draw->dcd_flags_1 = gfx->dcd.flags_1;
      return;
   }

   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;

   bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;
   bool writes_z = writes_depth(cmdbuf);
   bool writes_s = writes_stencil(cmdbuf);
   bool shader_modifies_coverage = false;
   uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                     MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
   uint8_t rt_written = 0, rt_read = 0;

   if (fs) {
      rt_written = color_attachment_written_mask(fs, &dyns->cal);
      rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
      shader_modifies_coverage = fs->info.fs.writes_coverage ||
                                 fs->info.fs.can_discard || alpha_to_coverage;
   }

   bool msaa = dyns->ms.rasterization_samples > 1;
   enum mesa_prim reduced_prim = u_reduced_prim(draw->info.prim);
   if (reduced_prim == MESA_PRIM_LINES &&
       rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM) {
      /* we need to disable MSAA when rendering bresenham lines.
       *
       * From the Vulkan spec:
       *   "When Bresenham lines are being rasterized, sample locations may
       *    all be treated as being at the pixel center (this may affect
       *    attribute and depth interpolation).""
       */
      msaa = false;
   }

   pan_pack(&draw->dcd_flags_0, DCD_FLAGS_0, cfg) {
      if (fs) {
         enum pan_earlyzs_zs_tilebuf_read zs_read =
            PAN_EARLYZS_ZS_TILEBUF_NOT_READ;

         if (z_attachment_read(fs, &dyns->ial) ||
             s_attachment_read(fs, &dyns->ial))
            zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT;

         bool roa_color =
            dyns->rasterization_order_access & VK_IMAGE_ASPECT_COLOR_BIT;

         cfg.allow_forward_pixel_to_kill =
            fs->info.fs.can_fpk && !(rt_mask & ~rt_written) &&
            !(rt_read & rt_written) && !alpha_to_coverage &&
            !cmdbuf->state.gfx.cb.info.any_dest_read && !roa_color;

         cfg.allow_forward_pixel_to_be_killed = !fs->info.writes_global;

         bool writes_zs = writes_z || writes_s;
         bool zs_always_passes = ds_test_always_passes(cmdbuf);
         bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                   MALI_OCCLUSION_MODE_DISABLED;

         bool roa_zs =
            dyns->rasterization_order_access &
            (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

         if (roa_zs)
            zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT;

         struct pan_earlyzs_state earlyzs =
            pan_earlyzs_get(fs->fs.earlyzs_lut, writes_zs || oq,
                            alpha_to_coverage, zs_always_passes, zs_read);

         cfg.pixel_kill_operation = (enum mali_pixel_kill)earlyzs.kill;
         cfg.zs_update_operation = (enum mali_pixel_kill)earlyzs.update;

         /* Use per-sample shading if required by API. Also use it when a
          * blend shader is used with multisampling, as this is handled by a
          * single ST_TILE in the blend shader with the current sample ID,
          * requiring per-sample shading.
          */
         cfg.evaluate_per_sample = (fs->info.fs.sample_shading ||
                                    cmdbuf->state.gfx.cb.info.needs_shader) &&
                                   (dyns->ms.rasterization_samples > 1);

         cfg.shader_modifies_coverage = shader_modifies_coverage;
      } else {
         cfg.allow_forward_pixel_to_kill = true;
         cfg.allow_forward_pixel_to_be_killed = true;
         cfg.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.overdraw_alpha0 = true;
         cfg.overdraw_alpha1 = true;
      }

      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.aligned_line_ends = true;

      cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;

      /* Face culling only applies to polygons: points and lines must not be
       * discarded by the front/back face cull bits. */
      const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;

      cfg.cull_front_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.cull_back_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      cfg.multisample_enable = msaa;
      cfg.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.alpha_to_coverage = alpha_to_coverage;
      cfg.scissor_to_bounding_box = true;
   }

   pan_pack(&draw->dcd_flags_1, DCD_FLAGS_1, cfg) {
      cfg.sample_mask = dyns->ms.sample_mask;
      cfg.render_target_mask = rt_written;
   }

   gfx->dcd.flags_valid = true;
   gfx->dcd.flags_0 = draw->dcd_flags_0;
   gfx->dcd.flags_1 = draw->dcd_flags_1;
}

static void
prepare_viewport(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   const VkViewport *viewport =
      &cmdbuf->vk.dynamic_graphics_state.vp.viewports[0];
   const VkRect2D *scissor = &cmdbuf->vk.dynamic_graphics_state.vp.scissors[0];

   pan_pack(&draw->scissor, SCISSOR, cfg) {
      /* The spec says "width must be greater than 0.0" */
      assert(viewport->width >= 0);
      int minx = (int)viewport->x;
      int maxx = (int)(viewport->x + viewport->width);

      /* Viewport height can be negative */
      int miny = MIN2((int)viewport->y, (int)(viewport->y + viewport->height));
      int maxy = MAX2((int)viewport->y, (int)(viewport->y + viewport->height));

      assert(scissor->offset.x >= 0 && scissor->offset.y >= 0);
      minx = MAX2(scissor->offset.x, minx);
      miny = MAX2(scissor->offset.y, miny);
      maxx = MIN2(scissor->offset.x + scissor->extent.width, maxx);
      maxy = MIN2(scissor->offset.y + scissor->extent.height, maxy);

      /* Make sure we don't end up with a max < min when width/height is 0 */
      maxx = maxx > minx ? maxx - 1 : maxx;
      maxy = maxy > miny ? maxy - 1 : maxy;

      /* Clamp viewport scissor to valid range */
      cfg.scissor_minimum_x = CLAMP(minx, 0, UINT16_MAX);
      cfg.scissor_minimum_y = CLAMP(miny, 0, UINT16_MAX);
      cfg.scissor_maximum_x = CLAMP(maxx, 0, UINT16_MAX);
      cfg.scissor_maximum_y = CLAMP(maxy, 0, UINT16_MAX);
   }

   /* Needs up-to-date viewport sysvals, so call after
    * cmd_prepare_draw_sysvals(). */
   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                     &draw->z_min, &draw->z_max);
}

static enum mali_index_type
index_size_to_index_type(uint32_t size)
{
   switch (size) {
   case 0:
      return MALI_INDEX_TYPE_NONE;
   case 1:
      return MALI_INDEX_TYPE_UINT8;
   case 2:
      return MALI_INDEX_TYPE_UINT16;
   case 4:
      return MALI_INDEX_TYPE_UINT32;
   default:
      UNREACHABLE("Invalid index size");
   }
}

static void
panvk_emit_primitive(struct panvk_cmd_buffer *cmdbuf,
                     const struct panvk_draw_data *draw, bool secondary_shader,
                     struct mali_primitive_packed *prim)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   bool writes_point_size =
      vs->info.vs.writes_point_size && draw->info.prim == MESA_PRIM_POINTS;
   bool writes_prim_id = vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID;
   bool fs_reads_prim_id = fs ? fs->info.fs.reads_primitive_id : false;

   /* v9 has no explicit position FIFO format: the extended FIFO (position
    * plus point size/layer slot) is selected through the point size array
    * format. The position shader writes gl_Layer (and the other extended
    * outputs) there even when not drawing points, so it must be enabled
    * whenever the shader needs it, otherwise each vertex overwrites the
    * position of the next one. */
   bool extended_fifo = writes_point_size || vs->info.vs.needs_extended_fifo;

   pan_pack(prim, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->info.prim);
      if (extended_fifo)
         cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP16;
      cfg.primitive_index_enable = fs_reads_prim_id;
      cfg.primitive_index_override = writes_prim_id && fs_reads_prim_id;

      /* Rotating primitives would change the provoking vertex used for flat
       * shading. The CSF backend never enables it either. */
      cfg.allow_rotating_primitives = false;

      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.secondary_shader = secondary_shader;

      cfg.index_type = index_size_to_index_type(draw->info.index.index_size);
      /* Vulkan always uses the maximum index value for restart, which is the
       * only kind of restart supported by v9. */
      cfg.primitive_restart =
         draw->info.index.index_size && draw->info.index.restart_enable;

      /* On Valhall, the base vertex offset is used for both indexed and
       * non-indexed draws: vertexOffset for the former, firstVertex for the
       * latter. Base instance is applied through the attribute descriptors
       * and sysvals, like on CSF. */
      cfg.base_vertex_offset = draw->info.vertex.base;
      cfg.instance_offset = 0;
      cfg.index_count = draw->info.vertex.count;
   }
}

static void
panvk_emit_shader_env(struct MALI_SHADER_ENVIRONMENT *cfg,
                      const struct panvk_shader_desc_state *desc,
                      uint64_t spd, uint64_t tls, uint64_t fau,
                      uint32_t fau_count)
{
   cfg->resources = desc->res_table;
   cfg->shader = spd;
   cfg->thread_storage = tls;
   cfg->fau = fau;
   cfg->fau_count = fau_count;
}

static VkResult
panvk_draw_prepare_malloc_vertex_job(struct panvk_cmd_buffer *cmdbuf,
                                     struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   /* Varying shaders only feed the fragment shader. */
   const bool secondary_shader = vs->info.vs.secondary_enable && fs != NULL;
   const bool points = draw->info.prim == MESA_PRIM_POINTS;

   struct pan_ptr ptr = panvk_cmd_alloc_desc(cmdbuf, MALLOC_VERTEX_JOB);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   util_dynarray_append(&batch->jobs, ptr.cpu);
   draw->job = ptr;

   panvk_emit_primitive(cmdbuf, draw, secondary_shader,
                        pan_section_ptr(ptr.cpu, MALLOC_VERTEX_JOB, PRIMITIVE));

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, INSTANCE_COUNT, cfg) {
      cfg.count = draw->info.instance.count;
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, ALLOCATION, cfg) {
      if (secondary_shader) {
         unsigned sz = vs->info.varyings.formats.generic_size_B;
         cfg.vertex_packet_stride = sz + 16;
         cfg.vertex_attribute_stride = sz;
      } else {
         /* Hardware requirement for "no varyings" */
         cfg.vertex_packet_stride = 16;
         cfg.vertex_attribute_stride = 0;
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, TILER, cfg) {
      cfg.address = draw->tiler_ctx->valhall.desc;
   }

   memcpy(pan_section_ptr(ptr.cpu, MALLOC_VERTEX_JOB, SCISSOR), &draw->scissor,
          pan_size(SCISSOR));

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, PRIMITIVE_SIZE, cfg) {
      if (vs->info.vs.writes_point_size && points) {
         /* The point size comes from the position shader. */
         cfg.size_array = 0;
      } else {
         /* If maintenance5 is enabled and a value is not written to a
          * variable decorated with PointSize, a value of 1.0 is used as the
          * size of points. */
         cfg.fixed_sized = points ? 1.0f : rs->line.width;
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, INDICES, cfg) {
      if (draw->info.index.index_size) {
         cfg.address = draw->info.index.buffer_dev_addr +
                       (uint64_t)draw->info.index.offset *
                          draw->info.index.index_size;
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, DRAW, cfg) {
      pan_unpack(&draw->dcd_flags_0, DCD_FLAGS_0, flags_0);
      pan_unpack(&draw->dcd_flags_1, DCD_FLAGS_1, flags_1);
      cfg.flags_0 = flags_0;
      cfg.flags_1 = flags_1;
      /* Varyings live in vertex packets. In memory-allocated IDVS mode, the
       * hardware writes the vertex array pointer and strides itself, but the
       * packet bit must be set (Gallium does the same). */
      cfg.vertex_array.packet = true;
      cfg.minimum_z = draw->z_min;
      cfg.maximum_z = draw->z_max;
      cfg.depth_stencil = draw->zsd;
      cfg.blend = draw->blend;
      cfg.blend_count = draw->blend_count;
      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;

      if (fs) {
         panvk_emit_shader_env(&cfg.shader, &cmdbuf->state.gfx.fs.desc,
                               panvk_priv_mem_dev_addr(fs->spd), draw->tls,
                               cmdbuf->state.gfx.fs.push_uniforms,
                               fs->fau.total_count);
      }
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, POSITION, cfg) {
      uint64_t pos_spd = points ? panvk_priv_mem_dev_addr(vs->spds.pos_points)
                                : panvk_priv_mem_dev_addr(vs->spds.pos_triangles);

      panvk_emit_shader_env(&cfg, &cmdbuf->state.gfx.vs.desc, pos_spd,
                            draw->tls, cmdbuf->state.gfx.vs.push_uniforms,
                            vs->fau.total_count);
   }

   pan_section_pack(ptr.cpu, MALLOC_VERTEX_JOB, VARYING, cfg) {
      /* The varying shader uses the same state as the position shader. */
      if (secondary_shader) {
         panvk_emit_shader_env(&cfg, &cmdbuf->state.gfx.vs.desc,
                               panvk_priv_mem_dev_addr(vs->spds.var),
                               draw->tls, cmdbuf->state.gfx.vs.push_uniforms,
                               vs->fau.total_count);
      }
   }

   return VK_SUCCESS;
}

static VkResult
prepare_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   VkResult result;

   /* On v9, the IDVS jobs allocate the vertex packets and the polygon lists
    * of the whole batch from the tiler heap, which is only released when the
    * fragment job runs. Unlike kbase, panfrost has no incremental rendering:
    * when the heap is exhausted, the tiler job fails with an OOM fault and
    * the draws are lost. Primitives scattered across the framebuffer use up
    * to ~64 bytes of heap each (2M random points exhaust the 128 MiB heap),
    * so split the batch, which spills and reloads the framebuffer, once it
    * holds this many vertices. Indirect draws are counted as one vertex. */
   const uint64_t max_batch_vertices = 1 << 20;

   /* This may split the batch, so do it before the job and vertex budget
    * checks below. */
   if (cmdbuf->state.gfx.vk_meta) {
      /* vk_meta doesn't care about the provoking vertex mode, we should use
       * the same mode that the application uses. */
      set_provoking_vertex_mode(cmdbuf, U_TRISTATE_UNSET);
   } else {
      enum u_tristate first_provoking_vertex = u_tristate_make(
         cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
         VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);
      set_provoking_vertex_mode(cmdbuf, first_provoking_vertex);
   }
   batch = cmdbuf->cur_batch;

   uint32_t draw_layers = cmdbuf->state.gfx.render.view_mask
                             ? util_bitcount(cmdbuf->state.gfx.render.view_mask)
                             : cmdbuf->state.gfx.render.layer_count;
   uint64_t draw_vertices = (uint64_t)draw->info.vertex.count *
                            draw->info.instance.count * MAX2(draw_layers, 1);

   /* There are only 16 bits in the descriptor for the job ID, and we need
    * one job per layer (two for indirect draws, which have a helper job). */
   if (batch->vtc_jc.job_index + 2 * cmdbuf->state.gfx.render.layer_count >=
          UINT16_MAX ||
       (batch->vertex_count &&
        batch->vertex_count + draw_vertices > max_batch_vertices)) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }
   batch->vertex_count += draw_vertices;

   ASSERTED const struct pan_fb_layout *fb =
      &cmdbuf->state.gfx.render.fb.layout;
   uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
   uint32_t rasterization_samples =
      cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

   /* If there's no attachment, and the FB descriptor hasn't been allocated
    * yet, we patch nr_samples to match rasterization_samples, otherwise, we
    * make sure those two numbers match. */
   if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments) {
      assert(rasterization_samples > 0);
      *nr_samples = rasterization_samples;
   } else {
      assert(rasterization_samples == *nr_samples);
   }

   assert(fb->sample_count == 0 ||
          fb->sample_count == cmdbuf->state.gfx.render.fb.nr_samples);

   result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   if (cmdbuf->state.gfx.vi.base_instance != draw->info.instance.base) {
      cmdbuf->state.gfx.vi.base_instance = draw->info.instance.base;
      gfx_state_set_dirty(cmdbuf, BASE_INSTANCE);
   }

   result = prepare_descs(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   draw->tls = batch->tls.gpu;
   draw->fb = batch->fb.desc.gpu;

   result = prepare_zsd(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   /* Must come before cmd_prepare_draw_sysvals(): it fills the blend
    * descriptor sysvals. */
   result = prepare_blend(cmdbuf, fs, draw);
   if (result != VK_SUCCESS)
      return result;

   prepare_dcd_flags(cmdbuf, fs, draw);

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, &draw->info, fs);

   prepare_viewport(cmdbuf, draw);

   /* Push uniforms (FAU) only change when the push constants, the shaders or
    * the sysvals they use change: cmd_prepare_draw_sysvals() flags them. */
   if (gfx_state_dirty(cmdbuf, VS_PUSH_UNIFORMS)) {
      struct pan_ptr vs_push_uniforms;
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
         cmdbuf, vs, &vs_push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.gfx.vs.push_uniforms = vs_push_uniforms.gpu;
   }

   if (fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, FS_PUSH_UNIFORMS)) {
      if (fs) {
         struct pan_ptr fs_push_uniforms;
         result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
            cmdbuf, fs, &fs_push_uniforms, 1);
         if (result != VK_SUCCESS)
            return result;
         cmdbuf->state.gfx.fs.push_uniforms = fs_push_uniforms.gpu;
      } else {
         cmdbuf->state.gfx.fs.push_uniforms = 0;
      }
   }

   return VK_SUCCESS;
}

/* Like on Bifrost, layered rendering and multiview issue one job per layer:
 * the vertex shader discards primitives written to other layers (and gets
 * the view index) through the layer_id sysval, so the VS push uniforms need
 * to be re-emitted per layer when the shader uses it. */
static VkResult
prepare_draw_layer(struct panvk_cmd_buffer *cmdbuf,
                   const struct panvk_shader_variant *vs,
                   struct panvk_draw_data *draw, uint32_t layer)
{
   draw->info.layer_id = layer;

   if (cmdbuf->state.gfx.sysvals.layer_id != (int32_t)layer) {
      cmdbuf->state.gfx.sysvals.layer_id = layer;

      if (shader_uses_sysval(vs, graphics, layer_id)) {
         struct pan_ptr push_uniforms;
         VkResult result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(
            cmdbuf, vs, &push_uniforms, 1);
         if (result != VK_SUCCESS)
            return result;
         cmdbuf->state.gfx.vs.push_uniforms = push_uniforms.gpu;
      }
   }

   return panvk_draw_prepare_tiler_context(cmdbuf, draw);
}

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->spds.pos_triangles))
      return;

   /* Memory-allocated IDVS is the only way to draw on Valhall with JM. */
   assert(vs->info.vs.idvs);

   /* With rasterizer discard, the only thing left would be vertex shader
    * side effects, and we don't expose vertexPipelineStoresAndAtomics on v9.
    * Keep the state dirty so the next draw picks it up. */
   if (cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable)
      return;

   /* Needs to be done before get_fs() is called because it depends on
    * fs.required being initialized. */
   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   /* Like on Bifrost, we issue one job per layer (or view), each with its
    * own tiler context. */
   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;

      result = prepare_draw_layer(cmdbuf, vs, draw, layer);
      if (result != VK_SUCCESS)
         return;

      result = panvk_draw_prepare_malloc_vertex_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return;

      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_MALLOC_VERTEX, false, false,
                     0, 0, &draw->job, false);
   }

   clear_dirty_after_draw(cmdbuf);
}

static uint64_t
vs_sysval_addr(const struct panvk_cmd_buffer *cmdbuf,
               const struct panvk_shader_variant *vs, uint32_t sysval_offset)
{
   /* Writes to unused sysvals go to the memory sink. */
   return cmdbuf->state.gfx.vs.push_uniforms +
          shader_remapped_sysval_offset(vs, sysval_offset);
}

static void
panvk_cmd_draw_indirect(struct panvk_cmd_buffer *cmdbuf,
                        struct panvk_draw_data *draw)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   VkResult result;

   /* If there's no vertex shader, we can skip the draw. */
   if (!panvk_priv_mem_check_alloc(vs->spds.pos_triangles))
      return;

   assert(vs->info.vs.idvs);
   assert(draw->info.indirect.buffer_dev_addr != 0);

   /* See panvk_cmd_draw(). */
   if (dyns->rs.rasterizer_discard_enable)
      return;

   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   /* The draw parameters are only known on the GPU. Emit the job and the
    * state with placeholders (no base vertex/instance, one vertex, one
    * instance), and let the indirect helper patch them. The vertex
    * attribute descriptors and the push uniforms are patched too, so make
    * sure fresh ones are allocated for this draw. */
   draw->info.vertex.base = 0;
   draw->info.vertex.count = 1;
   draw->info.instance.base = 0;
   draw->info.instance.count = 1;
   cmdbuf->state.gfx.vi.base_instance = 0;
   gfx_state_set_dirty(cmdbuf, BASE_INSTANCE);
   gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);

   result = prepare_draw(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return;

   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct vk_vertex_input_state *vi = dyns->vi;

   /* Per-instance attributes whose offset depends on the base instance. */
   const uint32_t attribs_mask =
      cmdbuf->state.gfx.vi.attribs_changing_on_base_instance;
   uint64_t attrib_strides = 0;
   if (attribs_mask) {
      struct pan_ptr strides = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, MAX_VS_ATTRIBS * sizeof(uint32_t), sizeof(uint32_t));
      if (!strides.gpu)
         return;

      uint32_t *s = strides.cpu;
      u_foreach_bit(i, attribs_mask)
         s[i] = dyns->vi_binding_strides[vi->attributes[i].binding];
      attrib_strides = strides.gpu;
   }

   STATIC_ASSERT(pan_size(ATTRIBUTE) == PANVK_DESCRIPTOR_SIZE);
   /* The vertex attributes are the first entries of the VS driver set. */
   const uint64_t attribs = cmdbuf->state.gfx.vs.desc.driver_set.dev_addr;

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   assert(view_mask == 0 || util_bitcount(view_mask) <= batch->fb.layer_count);
   uint32_t enabled_layer_count = view_mask
                                     ? util_bitcount(view_mask)
                                     : cmdbuf->state.gfx.render.layer_count;

   for (uint32_t i = 0; i < enabled_layer_count; i++) {
      const uint32_t layer = (view_mask != 0) ? u_bit_scan(&view_mask) : i;

      result = prepare_draw_layer(cmdbuf, vs, draw, layer);
      if (result != VK_SUCCESS)
         return;

      result = panvk_draw_prepare_malloc_vertex_job(cmdbuf, draw);
      if (result != VK_SUCCESS)
         return;

      /* The VS push uniforms may be different for each layer. */
      uint64_t first_vertex_sysval = 0x8ull << 60;
      uint64_t first_instance_sysval = 0x8ull << 60;
      if (shader_uses_sysval(vs, graphics, vs.first_vertex))
         first_vertex_sysval = vs_sysval_addr(
            cmdbuf, vs, sysval_offset(graphics, vs.first_vertex));
      if (shader_uses_sysval(vs, graphics, vs.base_instance))
         first_instance_sysval = vs_sysval_addr(
            cmdbuf, vs, sysval_offset(graphics, vs.base_instance));

      /* The attribute descriptors are shared by all layers: only patch
       * them once. */
      const uint32_t layer_attribs_mask = i == 0 ? attribs_mask : 0;

      if (draw->info.index.index_size) {
         const struct panlib_draw_indexed_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .job = draw->job.gpu,
            .index_buffer_ptr = draw->info.index.buffer_dev_addr,
            .attribs = attribs,
            .attrib_strides = attrib_strides,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .attribs_mask = layer_attribs_mask,
            .index_size = draw->info.index.index_size,
         };
         panlib_draw_indexed_indirect_helper_struct(
            &precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
            args);
      } else {
         const struct panlib_draw_indirect_helper_args args = {
            .cmd = draw->info.indirect.buffer_dev_addr,
            .job = draw->job.gpu,
            .attribs = attribs,
            .attrib_strides = attrib_strides,
            .first_vertex_sysval = first_vertex_sysval,
            .first_instance_sysval = first_instance_sysval,
            .attribs_mask = layer_attribs_mask,
         };
         panlib_draw_indirect_helper_struct(
            &precomp_ctx, panlib_1d(1), PANLIB_BARRIER_JM_SUPPRESS_PREFETCH,
            args);
      }

      /* The draw job depends on the helper job that patches it. */
      uint32_t helper_job = batch->vtc_jc.job_index;
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_MALLOC_VERTEX, false, false,
                     0, helper_job, &draw->job, false);
   }

   /* The attribute descriptors were patched with the indirect base
    * instance: make sure the next draw doesn't reuse them. */
   cmdbuf->state.gfx.vi.base_instance = UINT32_MAX;

   clear_dirty_after_draw(cmdbuf);

   /* Same for the VS push uniforms, whose sysvals were patched. */
   gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
}

#endif /* PAN_ARCH < 9 */

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                        uint32_t instanceCount, uint32_t firstVertex,
                        uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || vertexCount == 0)
      return;

   /* gl_BaseVertexARB is a signed integer, and it should expose the value of
    * firstVertex in a non-indexed draw. */
   assert(firstVertex < INT32_MAX);

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

   struct panvk_draw_data draw = {
      .info = {
         .vertex.base = firstVertex,
#if PAN_ARCH < 9
         .vertex.raw_offset = firstVertex,
#endif
         .vertex.count = vertexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
#if PAN_ARCH < 9
      .vertex_range = vertexCount,
      .padded_vertex_count =
         padded_vertex_count(cmdbuf, vertexCount, instanceCount),
#endif
   };

   panvk_cmd_draw(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexed)(VkCommandBuffer commandBuffer,
                               uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset,
                               uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || indexCount == 0)
      return;

   /* gl_BaseInstance is a signed integer, and it should expose the value of
    * firstInstnace. */
   assert(firstInstance < INT32_MAX);

#if PAN_ARCH >= 9
   /* IDVS doesn't need the min/max index search done on Bifrost, so indexed
    * draws can go through the direct path. */
   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, firstIndex),
         .vertex.base = vertexOffset,
         .vertex.count = indexCount,
         .instance.base = firstInstance,
         .instance.count = instanceCount,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw(cmdbuf, &draw);
#else
   struct pan_ptr indirect_index_alloc = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct VkDrawIndexedIndirectCommand), 8);

   struct VkDrawIndexedIndirectCommand *indirect_index_alloc_ptr =
      indirect_index_alloc.cpu;

   *indirect_index_alloc_ptr = (struct VkDrawIndexedIndirectCommand){
      .indexCount = indexCount,
      .instanceCount = instanceCount,
      .firstIndex = firstIndex,
      .vertexOffset = vertexOffset,
      .firstInstance = firstInstance,
   };

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, 0),
         .indirect.buffer_dev_addr = indirect_index_alloc.gpu,
         .indirect.draw_count = 1,
         .indirect.stride = 0,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
#endif
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                VkDeviceSize offset, uint32_t drawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   if (drawCount == 0)
      return;

   /* We cannot support arbitrary draw count on JM */
   assert(drawCount == 1);

   struct panvk_draw_data draw = {
      .info = {
         .indirect.buffer_dev_addr = panvk_buffer_gpu_ptr(buffer, offset),
         .indirect.draw_count = drawCount,
         .indirect.stride = stride,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer,
                                       VkBuffer _buffer, VkDeviceSize offset,
                                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   /* Because we don't currently advertise nullDescriptor for JM, it is only
    * valid to draw with a null index buffer if the draw accesses 0 indices.
    * For direct draws, this is covered by checks on instancedCount and
    * indexCount. For indirect draws we need to add an additional check, under
    * the assumption that if the index buffer is null, the draw must be empty.
    */
   if (drawCount == 0 || cmdbuf->state.gfx.ib.size == 0)
      return;

   /* We cannot support arbitrary draw count on JM */
   assert(drawCount == 1);

   struct panvk_draw_data draw = {
      .info = {
         .index = panvk_draw_info_index(cmdbuf, 0),
         .indirect.buffer_dev_addr = panvk_buffer_gpu_ptr(buffer, offset),
         .indirect.draw_count = drawCount,
         .indirect.stride = stride,
         .prim = panvk_get_client_prim(cmdbuf),
      },
   };

   panvk_cmd_draw_indirect(cmdbuf, &draw);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginRendering)(VkCommandBuffer commandBuffer,
                                  const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;
   bool resuming = pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT;

   /* When resuming from a suspended pass, the state should be unchanged. */
   if (resuming && cmdbuf->cur_batch) {
      state->render.flags = pRenderingInfo->flags;
   } else {
      /* If we're not resuming, cur_batch should be NULL.  However, this
       * currently isn't true because of how events are implemented.
       *
       * XXX: Rewrite events to not close and open batch and add an assert here.
       */
      if (cmdbuf->cur_batch)
         panvk_per_arch(cmd_close_batch)(cmdbuf);

      panvk_per_arch(cmd_init_render_state)(cmdbuf, pRenderingInfo);
      cmdbuf->state.gfx.render.fb.needs_load = !resuming;
   }

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndRendering2KHR)(
   VkCommandBuffer commandBuffer,
   UNUSED const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!(cmdbuf->state.gfx.render.flags & VK_RENDERING_SUSPENDING_BIT)) {
      const struct pan_fb_load *fb_load = &cmdbuf->state.gfx.render.fb.load;
      bool always_load = fb_load->z.always || fb_load->s.always;
      for (unsigned rt = 0; rt < PAN_MAX_RTS; rt++) {
         if (fb_load->rts[rt].always)
            always_load = true;
      }

      if (always_load)
         panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);

      cmdbuf->state.gfx.render.fb.needs_store = true;

      panvk_per_arch(cmd_close_batch)(cmdbuf);
      cmdbuf->cur_batch = NULL;
      panvk_per_arch(cmd_meta_resolve_attachments)(cmdbuf);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawMultiEXT)(VkCommandBuffer commandBuffer,
                                uint32_t drawCount,
                                const VkMultiDrawInfoEXT *pVertexInfo,
                                uint32_t instanceCount, uint32_t firstInstance,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   unsigned i = 0;

   vk_foreach_multi_draw(draw, i, pVertexInfo, drawCount, stride) {
      cmdbuf->state.gfx.draw_id = i;
      panvk_per_arch(CmdDraw)(commandBuffer, draw->vertexCount, instanceCount,
                              draw->firstVertex, firstInstance);
   }

   cmdbuf->state.gfx.draw_id = 0;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawMultiIndexedEXT)(
   VkCommandBuffer commandBuffer, uint32_t drawCount,
   const VkMultiDrawIndexedInfoEXT *pIndexInfo, uint32_t instanceCount,
   uint32_t firstInstance, uint32_t stride, const int32_t *pVertexOffset)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   unsigned i = 0;

   vk_foreach_multi_draw_indexed(draw, i, pIndexInfo, drawCount, stride) {
      cmdbuf->state.gfx.draw_id = i;
      panvk_per_arch(CmdDrawIndexed)(
         commandBuffer, draw->indexCount, instanceCount, draw->firstIndex,
         pVertexOffset ? *pVertexOffset : draw->vertexOffset, firstInstance);
   }

   cmdbuf->state.gfx.draw_id = 0;
}
