// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2023-2024 Second State INC

#pragma once

#include "plugin/plugin.h"
#include "types.h"
#include "common.h"
#include "llama.h"


namespace WasmEdge::Host::WASINN {
struct WasiNNEnvironment;
}

namespace WasmEdge::Host::WASINN::LlamaCpp {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_LLAMACPP
struct Graph {
    llama_model *LlamaModel = nullptr;
    gpt_params *params = nullptr;
};

struct Context {
    Context(size_t GId, Graph &) noexcept : GraphId(GId) {}
    llama_context *LlamaCtx = nullptr;
    gpt_params *params = nullptr;
    char *out = nullptr;
};
#else
struct Graph {};
struct Context {
  Context(size_t, Graph &) noexcept {}
};
#endif

struct Environ {};

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept;
Expect<WASINN::ErrNo> initExecCtx(WASINN::WasiNNEnvironment &Env,
                                  uint32_t GraphId,
                                  uint32_t &ContextId) noexcept;
Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               char* Prompt) noexcept;
Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                char* ResponseBuffer,
                                uint32_t &BytesWritten) noexcept;
Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept;
} // namespace WasmEdge::Host::WASINN::LlamaCpp