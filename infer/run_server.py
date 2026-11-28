# Description: Adds Qwen3-VL compatibility hooks to the llama.cpp inference server.
# Author: Dipesh Karmakar <dipesh.karmakar@gmail.com>
# Modifications:
# - f9dc837: Added the inference, tutor, retrieval, search, verifier, and image components.
# - 130fab8: Added the inference server and container setup.

import llama_cpp
import llama_cpp.server.app as server_app
import llama_cpp.server.model as server_model
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
from llama_cpp.server.__main__ import main

# Preserve the refusal field expected by OpenAI-compatible clients.
_orig_create_chat_completion = llama_cpp.Llama.create_chat_completion


def _safe_create_chat_completion(self, *args, **kwargs):
    res = _orig_create_chat_completion(self, *args, **kwargs)
    if isinstance(res, dict) and "choices" in res:
        for choice in res["choices"]:
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                choice["message"].setdefault("refusal", None)
    return res


llama_cpp.Llama.create_chat_completion = _safe_create_chat_completion

_BaseLlamaProxy = server_model.LlamaProxy
_orig_load = _BaseLlamaProxy.load_llama_from_model_settings


# Install the vision handler only while loading Qwen3-VL models.
class CustomLlamaProxy(_BaseLlamaProxy):
    @staticmethod
    def load_llama_from_model_settings(settings):
        if settings.chat_format == "qwen3-vl":
            assert settings.clip_model_path is not None, "clip model not found"
            handler = Qwen3VLChatHandler(
                clip_model_path=settings.clip_model_path,
                verbose=settings.verbose,
            )
            orig_llama = llama_cpp.Llama

            def _llama_with_handler(*args, **kwargs):
                kwargs["chat_handler"] = handler
                return orig_llama(*args, **kwargs)

            llama_cpp.Llama = _llama_with_handler
            try:
                return _orig_load(settings)
            finally:
                llama_cpp.Llama = orig_llama
        return _orig_load(settings)


server_model.LlamaProxy = CustomLlamaProxy
if hasattr(server_app, "LlamaProxy"):
    server_app.LlamaProxy = CustomLlamaProxy

if __name__ == "__main__":
    main()
