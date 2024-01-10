/*!
 *  Copyright (c) 2023 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Huggingface tokenizer
 */
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace tokenizers {
/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
class TikToken : public Tokenizer {
 public:
  explicit TikToken(TiktokenHandle handle) : handle_(handle) {}

  TikToken(const TikToken&) = delete;
  TikToken(TikToken&& other) { std::swap(other.handle_, handle_); }

  ~TikToken() {
    if (handle_ != nullptr) {
      tiktoken_free(handle_);
    }
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string& text) final {
    const std::string& allowed_special = "[]";
    const uint32_t* out_data;
    size_t out_len;
    tiktoken_encode(
      handle_,
      text.data(), text.length(),
      allowed_special.data(), allowed_special.length(),
      &out_data, &out_len);
    return {out_data, out_data + out_len};
  }

  // use i32 to be consistent with sentencepiece
  std::string Decode(const std::vector<int32_t>& ids) final {
    bool skip_special_token = false;

    const char* out_cstr;
    size_t out_len;

    tiktoken_decode(
      handle_,
      reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
      &out_cstr, &out_len);
    return {out_cstr, out_len};
  }

 private:
  // internal handle
  TiktokenHandle handle_{nullptr};
};

std::unique_ptr<Tokenizer> Tokenizer::FromTiktoken(const std::string& json) {
  return std::make_unique<TikToken>(tiktoken_new_from_str(json.data(), json.length()));
}
}  // namespace tokenizers
