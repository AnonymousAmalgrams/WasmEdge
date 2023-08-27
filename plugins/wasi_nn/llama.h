#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#define LLAMA_MAX_DEVICES GGML_CUDA_MAX_DEVICES
#else
#define LLAMA_MAX_DEVICES 1
#endif // GGML_USE_CUBLAS
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_FILE_MAGIC_GGML        0x67676d6cu // 'ggml'

#define LLAMA_FILE_VERSION           3
#define LLAMA_FILE_MAGIC             LLAMA_FILE_MAGIC_GGJT
#define LLAMA_FILE_MAGIC_UNVERSIONED LLAMA_FILE_MAGIC_GGML
#define LLAMA_SESSION_MAGIC          LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION        1

#define LLAMA_DEFAULT_SEED           0xFFFFFFFF

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST) || defined(GGML_USE_METAL)
// Defined when llama.cpp is compiled with support for offloading model layers to GPU.
#define LLAMA_SUPPORTS_GPU_OFFLOAD
#endif

#ifndef LLAMA_DEFAULT_RMS_EPS
#define LLAMA_DEFAULT_RMS_EPS 5e-6f
#endif

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_model;
    struct llama_context;

    typedef int llama_token;

    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        llama_token_data * data;
        size_t size;
        bool sorted;
    } llama_token_data_array;

    typedef void (*llama_progress_callback)(float progress, void *ctx);

    enum llama_log_level {
        LLAMA_LOG_LEVEL_ERROR = 2,
        LLAMA_LOG_LEVEL_WARN  = 3,
        LLAMA_LOG_LEVEL_INFO  = 4
    };

    // Signature for logging events
    // Note that text includes the new line character at the end for most events.
    // If your logging mechanism cannot handle that, check if the last character is '\n' and strip it
    // if it exists.
    // It might not exist for progress report where '.' is output repeatedly.
    typedef void (*llama_log_callback)(enum llama_log_level level, const char * text, void * user_data);

    struct llama_context_params {
        uint32_t seed;         // RNG seed, -1 for random
        int32_t  n_ctx;        // text context
        int32_t  n_batch;      // prompt processing batch size
        int32_t  n_gqa;        // grouped-query attention (TEMP - will be moved to model hparams)
        float    rms_norm_eps; // rms norm epsilon (TEMP - will be moved to model hparams)
        int32_t  n_gpu_layers; // number of layers to store in VRAM
        int32_t  main_gpu;     // the GPU that is used for scratch and small tensors

        const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        float    rope_freq_base;  // RoPE base frequency
        float    rope_freq_scale; // RoPE frequency scaling factor

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool low_vram;   // if true, reduce VRAM usage at the cost of performance
        bool mul_mat_q;  // if true, use experimental mul_mat_q kernels
        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the llama_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only
    };
    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5, // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6, // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,// except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18,// except 1d tensors
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int nthread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype   ftype;    // quantize to this llama_ftype
        bool allow_requantize;       // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor; // quantize output.weight
    } llama_model_quantize_params;

    // grammar types
    struct llama_grammar;

    // grammar element type
    enum llama_gretype {
        // end of rule definition
        LLAMA_GRETYPE_END            = 0,

        // start of alternate definition for rule
        LLAMA_GRETYPE_ALT            = 1,

        // non-terminal element: reference to rule
        LLAMA_GRETYPE_RULE_REF       = 2,

        // terminal element: character (code point)
        LLAMA_GRETYPE_CHAR           = 3,

        // inverse char(s) ([^a], [^a-b] [^abc])
        LLAMA_GRETYPE_CHAR_NOT       = 4,

        // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
        // be an inclusive range ([a-z])
        LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

        // modifies a preceding LLAMA_GRETYPE_CHAR or
        // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
        LLAMA_GRETYPE_CHAR_ALT       = 6,
    };

    typedef struct llama_grammar_element {
        enum llama_gretype type;
        uint32_t           value; // Unicode code point or rule ID
    } llama_grammar_element;

    // performance timing information
    struct llama_timings {
        double t_start_ms;
        double t_end_ms;
        double t_load_ms;
        double t_sample_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_sample;
        int32_t n_p_eval;
        int32_t n_eval;
    };

    LLAMA_API struct llama_context_params llama_context_default_params();
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params();

    // TODO: not great API - very likely to change
    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(bool numa);
    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free();

    LLAMA_API int64_t llama_time_us();

    LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
            struct llama_context_params   params);

    LLAMA_API void llama_free_model(struct llama_model * model);

    LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    LLAMA_API DEPRECATED(struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params),
            "please use llama_load_model_from_file combined with llama_new_context_with_model instead");


    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    LLAMA_API DEPRECATED(int llama_apply_lora_from_file(
            struct llama_context * ctx,
                      const char * path_lora,
                      const char * path_base_model,
                             int   n_threads),
            "please use llama_model_apply_lora_from_file instead");

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    LLAMA_API int llama_eval(
            struct llama_context * ctx,
               const llama_token * tokens,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Same as llama_eval, but use float matrix input directly.
    LLAMA_API int llama_eval_embd(
            struct llama_context * ctx,
                     const float * embd,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    LLAMA_API int llama_tokenize(
            struct llama_context * ctx,
                      const char * text,
                     llama_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    LLAMA_API int llama_tokenize_with_model(
        const struct llama_model * model,
                      const char * text,
                     llama_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    LLAMA_API int llama_n_vocab(const struct llama_context * ctx);
    LLAMA_API int llama_n_ctx  (const struct llama_context * ctx);
    LLAMA_API int llama_n_embd (const struct llama_context * ctx);

    LLAMA_API int llama_n_vocab_from_model(const struct llama_model * model);
    LLAMA_API int llama_n_ctx_from_model  (const struct llama_model * model);
    LLAMA_API int llama_n_embd_from_model (const struct llama_model * model);

    // Get the vocabulary as output parameters.
    // Returns number of results.
    LLAMA_API int llama_get_vocab(
            const struct llama_context * ctx,
                          const char * * strings,
                                 float * scores,
                                   int   capacity);

    LLAMA_API int llama_get_vocab_from_model(
              const struct llama_model * model,
                          const char * * strings,
                                 float * scores,
                                   int   capacity);

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    LLAMA_API const char * llama_token_to_str(
            const struct llama_context * ctx,
                           llama_token   token);

    LLAMA_API const char * llama_token_to_str_with_model(
              const struct llama_model * model,
                           llama_token   token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos();  // beginning-of-sentence
    LLAMA_API llama_token llama_token_eos();  // end-of-sentence
    LLAMA_API llama_token llama_token_nl();   // next-line

    // Grammar
    //
    LLAMA_API struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index);

    LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);

    // Sampling functions

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    LLAMA_API void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty);

    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    LLAMA_API void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);

    /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
    /// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    LLAMA_API void llama_sample_classifier_free_guidance(
              struct llama_context * ctx,
            llama_token_data_array * candidates,
              struct llama_context * guidance_ctx,
                             float   scale);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_API void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    LLAMA_API void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
    LLAMA_API void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates, float temp);

    /// @details Apply constraints from grammar
    LLAMA_API void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar);

    /// @details Selects the token with the highest probability.
    LLAMA_API llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities.
    LLAMA_API llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates);

    /// @details Accepts the sampled token into the grammar
    LLAMA_API void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token);
#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_API_INTERNAL

#include <vector>
#include <string>
struct ggml_tensor;

const std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(struct llama_context * ctx);

#endif

#endif // LLAMA_H