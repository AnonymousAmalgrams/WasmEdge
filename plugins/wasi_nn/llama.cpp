//note: this is a lightly modified version of ggerganov's llama.cpp
// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "llama.h"

#include "ggml.h"
#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_MPI
#include "ggml-mpi.h"
#endif
#ifdef GGML_USE_K_QUANTS
#ifndef QK_K
#ifdef GGML_QKK_64
#define QK_K 64
#else
#define QK_K 256
#endif
#endif
#endif

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void llama_log_internal(llama_log_level level, const char* format, ...);
static void llama_log_callback_default(llama_log_level level, const char * text, void * user_data);
#define LLAMA_LOG_INFO(...)  llama_log_internal(LLAMA_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(LLAMA_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(LLAMA_LOG_LEVEL_ERROR, __VA_ARGS__)


#if !defined(GGML_USE_CUBLAS) && !defined(GGML_USE_METAL)
#include "ggml-alloc.h"
#define LLAMA_USE_ALLOCATOR
#else
#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16
#endif

// default hparams (LLaMA 7B)
struct llama_hparams {
    uint32_t n_vocab   = 32000;
    uint32_t n_ctx     = 512;   // this is provided as user input?
    uint32_t n_embd    = 4096;
    uint32_t n_mult    = 256;
    uint32_t n_head    = 32;
    uint32_t n_head_kv = 32;
    uint32_t n_layer   = 32;
    uint32_t n_rot     = 64;

    // LLaMAv2
    // TODO: load from model data hparams
    float f_ffn_mult = 1.0f;
    float f_rms_norm_eps = LLAMA_DEFAULT_RMS_EPS;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;

    bool operator!=(const llama_hparams & other) const {
        return static_cast<bool>(memcmp(this, &other, sizeof(llama_hparams))); // NOLINT
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    size_t kv_size() const {
        size_t result = 2ull;
        result *= (size_t) n_embd_gqa();
        result *= (size_t) n_ctx;
        result *= (size_t) n_layer;
        result *= sizeof(ggml_fp16_t);
        return result;
    }
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_kv_cache {
    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_model {
    e_model type = MODEL_UNKNOWN;

    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;
    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_ctx_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    llama_vocab vocab;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

struct llama_context {
    llama_context(const llama_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~llama_context() {
        if (model_owner) {
            delete &model;
        }
#ifdef GGML_USE_METAL
        if (ctx_metal) {
            ggml_metal_free(ctx_metal);
        }
#endif
#ifdef LLAMA_USE_ALLOCATOR
        if (alloc) {
            ggml_allocr_free(alloc);
        }
#endif
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_model & model;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    llama_ctx_buffer buf_compute;

#ifdef LLAMA_USE_ALLOCATOR
    llama_ctx_buffer buf_alloc;
    ggml_allocr * alloc = NULL;
#endif

#ifdef LLAMA_USE_SCRATCH
    llama_ctx_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];
    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }
    
    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};



struct llama_model * llama_load_model_from_file(
            struct llama_context_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!llama_model_load(*model, model->vocab, params.n_ctx, params.n_batch, params.n_gqa, params.rms_norm_eps, params.n_gpu_layers,
                params.main_gpu, params.tensor_split, params.mul_mat_q, params.rope_freq_base, params.rope_freq_scale,params.low_vram,
                memory_type, params.use_mmap, params.use_mlock, params.vocab_only, params.progress_callback,
                params.progress_callback_user_data)) {
        LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

static bool llama_model_load(
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gqa,
        float rms_norm_eps,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_model_load_internal(model, vocab, n_ctx, n_batch, n_gqa, rms_norm_eps, n_gpu_layers,
                                  main_gpu, tensor_split, mul_mat_q, rope_freq_base, rope_freq_scale, low_vram, memory_type,
                                  use_mmap, use_mlock, vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading model: %s\n", err.what());
        return false;
    }
}

static void llama_model_load_internal(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gqa,
        float rms_norm_eps,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {

    std::unique_ptr<llama_model_loader> ml(new llama_model_loader(fname, use_mmap));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;
    model.n_gpu_layers = n_gpu_layers;

    auto & hparams = model.hparams;

    // TODO: read from file
    hparams.f_rms_norm_eps = rms_norm_eps;

    {
        switch (hparams.n_layer) {
            case 26: model.type = e_model::MODEL_3B; break;
            case 32: model.type = e_model::MODEL_7B; break;
            case 40: model.type = e_model::MODEL_13B; break;
            case 60: model.type = e_model::MODEL_30B; break;
            case 80: model.type = e_model::MODEL_65B; break;
            default:
                {
                    if (hparams.n_layer < 32) {
                        model.type = e_model::MODEL_7B;
                    }
                } break;
        }

        hparams.n_ctx = n_ctx;

        // LLaMAv2
        // TODO: temporary until GGUF
        LLAMA_ASSERT(hparams.n_head % n_gqa == 0);
        hparams.n_head_kv = hparams.n_head / n_gqa;
        if (model.type == e_model::MODEL_65B && n_gqa == 8) {
            LLAMA_LOG_WARN("%s: warning: assuming 70B model based on GQA == %d\n", __func__, n_gqa);
            model.type = e_model::MODEL_70B;
            hparams.f_ffn_mult = 1.3f; // from the params.json of the 70B model
        }

        hparams.rope_freq_base  = rope_freq_base;
        hparams.rope_freq_scale = rope_freq_scale;
    }

    // ref: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L194-L199
    const uint32_t n_ff_raw  = 2*(4*hparams.n_embd)/3;
    const uint32_t n_ff_mult = hparams.f_ffn_mult*n_ff_raw;
    const uint32_t n_ff      = ((n_ff_mult + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init   (model.buf.addr);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format("ggml_init() failed"));
        }
    }

    (void) main_gpu;
    (void) mul_mat_q;
#if defined(GGML_USE_CUBLAS)
    LLAMA_LOG_INFO("%s: using CUDA for GPU acceleration\n", __func__);
    ggml_cuda_set_main_device(main_gpu);
    ggml_cuda_set_mul_mat_q(mul_mat_q);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
    LLAMA_LOG_INFO("%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU
#else
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

    // prepare memory for the weights
    size_t vram_weights = 0;
    size_t vram_scratch = 0;
    {
        const uint32_t n_embd     = hparams.n_embd;
        const uint32_t n_embd_gqa = hparams.n_embd_gqa();
        const uint32_t n_layer    = hparams.n_layer;
        const uint32_t n_vocab    = hparams.n_vocab;

        ml->ggml_ctx = ctx;

        model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab}, GGML_BACKEND_CPU);

        // "output" tensor
        {
            ggml_backend backend_norm;
            ggml_backend backend_output;
            if (n_gpu_layers > int(n_layer)) { // NOLINT
                // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                backend_norm = low_vram ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#else
                backend_norm = low_vram || n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
            } else {
                backend_norm = GGML_BACKEND_CPU;
                backend_output = GGML_BACKEND_CPU;
            }

            model.norm   = ml->get_tensor("norm.weight",   {n_embd},          backend_norm);
            model.output = ml->get_tensor("output.weight", {n_embd, n_vocab}, backend_output);
            if (backend_norm == GGML_BACKEND_GPU) {
                vram_weights += ggml_nbytes(model.norm);
            }
            if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                vram_weights += ggml_nbytes(model.output);
            }
        }

        const int i_gpu_start = n_layer - n_gpu_layers;

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            const ggml_backend backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
            const ggml_backend backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i);

            layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd}, backend);

            layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd},     backend_split);
            layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd},     backend_split);

            layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd}, backend);

            layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff}, backend_split);
            layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {  n_ff, n_embd}, backend_split);
            layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff}, backend_split);

            if (backend == GGML_BACKEND_GPU) {
                vram_weights +=
                    ggml_nbytes(layer.attention_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)             +
                    ggml_nbytes(layer.wv)             + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                    ggml_nbytes(layer.w1)             + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
            }
        }
    }

    ml->done_getting_tensors();

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights; // weights in VRAM not in memory

#ifndef LLAMA_USE_ALLOCATOR
        mem_required +=
            MEM_REQ_SCRATCH0(hparams.n_ctx).at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at(model.type);
#endif

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*hparams.kv_size();

        LLAMA_LOG_INFO("%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

        (void) vram_scratch;
        (void) n_batch;
#ifdef GGML_USE_CUBLAS
        if (low_vram) {
            LLAMA_LOG_INFO("%s: not allocating a VRAM scratch buffer due to low VRAM option\n", __func__);
            ggml_cuda_set_scratch_size(0); // disable scratch
        } else {
            const size_t vram_scratch_base = VRAM_REQ_SCRATCH_BASE().at(model.type);
            const size_t vram_scratch_per_context = VRAM_REQ_SCRATCH_PER_CONTEXT().at(model.type);
            vram_scratch = n_batch * (vram_scratch_base + n_ctx * vram_scratch_per_context);
            ggml_cuda_set_scratch_size(vram_scratch);
            if (n_gpu_layers > 0) {
                LLAMA_LOG_INFO("%s: allocating batch_size x (%zd kB + n_ctx x %zd B) = %zd MB VRAM for the scratch buffer\n",
                        __func__, vram_scratch_base / kB, vram_scratch_per_context,
                        (vram_scratch + MB - 1) / MB); // round up
            }
        }
#endif // GGML_USE_CUBLAS

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }
        size_t vram_kv_cache = 0;

#ifdef GGML_USE_CUBLAS
        const int max_backend_supported_layers = hparams.n_layer + 3;
        const int max_offloadable_layers = low_vram ? hparams.n_layer + 1 : hparams.n_layer + 3;
        if (n_gpu_layers > (int) hparams.n_layer + 1) {
            if (low_vram) {
                LLAMA_LOG_INFO("%s: cannot offload v cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
        if (n_gpu_layers > (int) hparams.n_layer + 2) {
            if (low_vram) {
                LLAMA_LOG_WARN("%s: cannot offload k cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
#elif defined(GGML_USE_CLBLAST)
        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;
#endif // GGML_USE_CUBLAS

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n",
                __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
        LLAMA_LOG_INFO("%s: total VRAM used: %zu MB\n",
                __func__, (vram_weights + vram_scratch + vram_kv_cache + MB - 1) / MB); // round up
#else
        (void) n_gpu_layers;
#endif // defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    }

    // populate `tensors_by_name`
    for (llama_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    (void) tensor_split;
#if defined(GGML_USE_CUBLAS)
    {
        ggml_cuda_set_tensor_split(tensor_split);
    }
#endif

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml->mapping);
}



struct llama_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<llama_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

struct llama_file_loader { 
    Span<const Span<uint8_t>> file;
    llama_hparams hparams;
    llama_vocab vocab;

    int outerIndex;
    int innerIndex;

    uint32_t read_u32() {
      uint32_t ret;
      int counter = 0;

      while(counter < 4)
      {
        ret = (ret << 8 ) | file[innerInex][outerIndex];
        if(innerIndex < file[outerIndex].size())
          innerIndex++;
        else
          outerIndex++;
        counter++;
      }

      return ret;
    }

    llama_file_loader(Span<const Span<uint8_t>> Builders, llama_load_tensors_map & tensors_map) {
      file = Builders;
      innerIndex = 0;
      outerIndex = 0;

      spdlog::error("llama.cpp: loading model from provided byte sequence");
      read_magic();
      read_hparams();
      read_vocab();
      read_tensor_metadata(tensors_map);
    }

    void read_magic() {
      uint32_t magic = read_u32();

      if (magic == LLAMA_FILE_MAGIC_GGML) {
        file_version = LLAMA_FILE_VERSION_GGML;
        return;
      }
      else
      {
        spdlog::error("unrecognized magic number: %08x, check ggml version", magic);
      }
    }

    void read_hparams() {
        hparams.n_vocab = read_u32();
        hparams.n_embd = read_u32();
        hparams.n_mult = read_u32();
        hparams.n_head = read_u32();
        hparams.n_layer = read_u32();
        hparams.n_rot = read_u32();
        hparams.ftype = (enum llama_ftype) read_u32();
    }

//TODO
    void read_vocab() { 
        vocab.id_to_token.resize(hparams.n_vocab);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            uint32_t len = read_u32();
            //std::string word = file.read_string(len);

            float score = 0.0f;
            //file.read_raw(&score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            //tok_score.tok = std::move(word);
            tok_score.score = score;
        }
    }
//TODO
void read_tensor_metadata(llama_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            llama_load_tensor tensor;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            tensor.type = (enum ggml_type) file.read_u32();
            tensor.ne.resize(n_dims);
            //file.read_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * n_dims);
            //std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw std::runtime_error(format("llama.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims));
            }
            switch (tensor.type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    break;
                default: {
                    throw std::runtime_error(format("unrecognized tensor type %u\n", tensor.type));
                }
            }

            // skip to the next multiple of 32 bytes
            //file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);

            tensor.file_off = file.tell();
            tensor.name = name;
            tensor.size = llama_calc_tensor_size(tensor.ne, tensor.type);
            //file.seek(tensor.size, SEEK_CUR);

            tensors_map.tensors.push_back(tensor);
            tensors_map.name_to_idx[name] = tensors_map.tensors.size() - 1;
        }
    }
}