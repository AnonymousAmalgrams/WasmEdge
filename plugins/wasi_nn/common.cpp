//note: this is a lightly modified version of ggerganov's llama.cpp/examples/common.cpp
#include "common.h"

#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <regex>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
  gpt_params defaualt_params;
  return true;
}

// TODO: not great allocating this every time
std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int) add_bos);
    const int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 4096,
        /*.n_batch                     =*/ 512,
        /*.n_gqa                       =*/ 1,
        /*.rms_norm_eps                =*/ LLAMA_DEFAULT_RMS_EPS,
        /*.gpu_layers                  =*/ 0,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rope_freq_base              =*/ 10000.0f,
        /*.rope_freq_scale             =*/ 1.0f,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.low_vram                    =*/ false,
        /*.mul_mat_q                   =*/ false,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
    };

    return result;
}

struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_default_params();

    lparams.n_ctx           = params.n_ctx;
    lparams.n_batch         = params.n_batch;
    lparams.n_gqa           = params.n_gqa;
    lparams.rms_norm_eps    = params.rms_norm_eps;
    lparams.n_gpu_layers    = params.n_gpu_layers;
    lparams.main_gpu        = params.main_gpu;
    lparams.tensor_split    = params.tensor_split;
    lparams.low_vram        = params.low_vram;
    lparams.mul_mat_q       = params.mul_mat_q;
    lparams.seed            = params.seed;
    lparams.f16_kv          = params.memory_f16;
    lparams.use_mmap        = params.use_mmap;
    lparams.use_mlock       = params.use_mlock;
    lparams.logits_all      = params.perplexity;
    lparams.embedding       = params.embedding;
    lparams.rope_freq_base  = params.rope_freq_base;
    lparams.rope_freq_scale = params.rope_freq_scale;

    return lparams;
}

std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_params_from_gpt_params(params);

    llama_model * model  = llama_load_model_from_file(lparams);
    if (model == NULL) {
        return std::make_tuple(nullptr, nullptr);
    }

    llama_context * lctx = llama_new_context_with_model(model, lparams);
    if (lctx == NULL) {
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    if (!params.lora_adapter.empty()) {
        int err = llama_model_apply_lora_from_file(model,
                                             params.lora_adapter.c_str(),
                                             params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr);
        }
    }

    return std::make_tuple(model, lctx);
}