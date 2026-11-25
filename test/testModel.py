import glob
from llama_cpp import Llama

path = glob.glob('/home/basilisk/Project_EduAi/Source/EduAi/models/qwen3-14b/*Q4_K_M*.gguf')[0]

llm = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, flash_attn=True, verbose=False)

# out = llm.create_chat_completion(messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}], max_tokens=64,)
out = llm.create_chat_completion(messages=[{'role': 'user', 'content': 'Who is Christopher Columbus? Kindly reply in 2 lines, no additional response and no advertisement please, I just need the answer from a third person perspective.'}])

print(out['choices'][0]['message']['content'])
