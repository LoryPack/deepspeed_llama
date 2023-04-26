from deepspeed_llama.models.llama import get_llama_hf_model,LlamaModel
import deepspeed
from transformers import pipeline
import time
from accelerate import Accelerator
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from transformers import AutoModel, AutoTokenizer

model_path = 'experiments/example_sweeps/sweep_configs/489760_0_results'
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModel.from_pretrained(model_path)
#exit()

#accelerator = Accelerator()

start_time = time.time()

model, tokenizer = get_llama_hf_model('llama-7b')

#model = load_state_dict_from_zero_checkpoint(model,model_path)

#model.load_state_dict(torch.load('test_state_dict'))

#model = LlamaForCausalLM.from_pretrained('experiments/example_sweeps/sweep_configs/485145_0_results', torch_dtype=torch.bfloat16, use_cache=False)
#tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, use_cache=False)
#tokenizer.pad_token_id = 0
#tokenizer.pad_token = tokenizer.decode(0)

#model, tokenizer = accelerator.prepare(model, tokenizer)

#model = accelerator.prepare(model,train_micro_batch_size_per_gpu=1)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer,device='cuda:0')#device=accelerator.device)

#generator.model = deepspeed.init_inference(generator.model,
#                                mp_size=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time to load model: {elapsed_time:.4f} seconds")

device = generator.model.device
print(f"The pipeline is running on device: {device}")

start_time = time.time()

input_text = "Once upon a time..."

output = generator(input_text,max_new_tokens=100)
print(output)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time 1 completion: {elapsed_time:.4f} seconds")

start_time = time.time()

prompts = ['At the end of the day, when he reflected, it was clear that',
           'Unicorns have not been sighted in the west-midlands for 400 years, and yet',
           '<prompt 1>',
           '<prompt 2>',
           '<prompt 3>',
           '<prompt 5>',
           '<prompt 5>']


output = generator(prompts,max_new_tokens=100)
print(output)



print(len(output))
for out in output:
    print(out)
    print(out[0]['generated_text'])

response = {'choices':[{'text':out[0]['generated_text']} for out in output]}

print(response["choices"][3]["text"].strip().lower())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time 2 completion: {elapsed_time:.4f} seconds")

exit()

"""
model = LlamaModel('llama-13b')


prompts = ['At the end of the day, when he reflected, it was clear that',
           'Unicorns have not been sighted in the west-midlands for 400 years, and yet']

print(prompts)
completions = model.generate(prompts,200)

print(completions)
"""
