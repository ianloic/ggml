// Slow Learner's own additions to ggml-vulkan, which aren't meant for
// upstream. They're kept in this file, reached through the backend's
// get_proc_address, so that the upstream files change by only a line or two
// and the fork rebases onto new ggml releases easily.

#include "ggml-vulkan-common.h"

#include <cstring>

// Whether FLASH_ATTN_EXT `op` would run on `dev` with cooperative matrices
// (coopmat1 or coopmat2) rather than the scalar shader, which on GPUs
// without them can be slower than the same attention as plain products and
// a softmax. It makes the choice ggml_vk_flash_attn() makes, for K and V of
// F16 or F32 without grouped queries, as Slow Learner's models have.
static bool ggml_backend_vk_flash_attn_uses_matrix_cores(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (dev == nullptr || op == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const auto * dev_ctx = static_cast<const ggml_backend_vk_device_context *>(dev->context);
    const vk_device device = ggml_vk_get_device(dev_ctx->device);
    const bool f32acc = !device->fp16 || op->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;
    const vk_fa_tuning_params params = get_fa_tuning_params(
        device, (uint32_t) k->ne[0], (uint32_t) v->ne[0], (uint32_t) q->ne[1], (uint32_t) k->ne[1], k->type, v->type, f32acc);
    return params.path != FA_SCALAR;
}

void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t /*reg*/, const char * name) {
    if (std::strcmp(name, "ggml_backend_vk_flash_attn_uses_matrix_cores") == 0) {
        return (void *) ggml_backend_vk_flash_attn_uses_matrix_cores;
    }
    return nullptr;
}
