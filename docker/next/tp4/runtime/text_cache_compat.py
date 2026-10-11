from vllm.config import VllmConfig
_original=VllmConfig._resolve_mm_embedding_inputs
def resolve(self):
 model=self.model_config
 mm=model.multimodal_config if model is not None else None
 if model is not None and getattr(model.hf_config,'model_type',None)=='qwen4_exp' and mm is not None and mm.language_model_only:
  mm.allow_missing_mm_embeddings=False
  mm.enable_mm_embeds=False
  return
 return _original(self)
VllmConfig._resolve_mm_embedding_inputs=resolve
