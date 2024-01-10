/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_c.h
 * \brief C binding to tokenizers rust library
 */
#ifndef TOKENIZERS_C_H_
#define TOKENIZERS_C_H_

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);

TokenizerHandle byte_level_bpe_tokenizers_new_from_str(const char* vocab, size_t vocab_len,
                                                       const char* merges, size_t merges_len,
                                                       const char* added_tokens,
                                                       size_t added_tokens_len);

void tokenizers_encode(TokenizerHandle handle, const char* data, size_t len, int add_special_token);

void tokenizers_decode(TokenizerHandle handle, const uint32_t* data, size_t len,
                       int skip_special_token);

void tokenizers_get_decode_str(TokenizerHandle handle, const char** data, size_t* len);

void tokenizers_get_encode_ids(TokenizerHandle handle, const uint32_t** id_data, size_t* len);

void tokenizers_free(TokenizerHandle handle);

typedef void* TiktokenHandle;

TiktokenHandle tiktoken_new_from_str(const char* json, size_t len);

void tiktoken_encode(
  TiktokenHandle handle,
  const char* input_cstr, size_t input_cstr_len,
  const char* allowed_special, size_t allowed_special_len,
  const uint32_t** out_data, size_t* out_len);

void tiktoken_decode(
  TiktokenHandle handle,
  const uint32_t* input_ids, size_t input_len,
  const char** out_cstr, size_t* out_len);

void tiktoken_free(TiktokenHandle handle);

#ifdef __cplusplus
}
#endif
#endif  // TOKENIZERS_C_H_
