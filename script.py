import torch
from fastapi import FastAPI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

app: FastAPI = FastAPI()

model_name: str = "Vikhrmodels/it-5.4-fp16-orpo-v2"
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="sequential",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)


@app.get("/")
async def generate_response(prompt: str = "Что ты такое?"):
    test_input: torch.Tensor = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    test_input = test_input.to(model.device)

    answer: torch.Tensor = model.generate(
        test_input,
        do_sample=True,
        use_cache=True,
        max_new_tokens=256,
        temperature=0.3,
    )[:, test_input.shape[-1] :]
    answer: str = tokenizer.batch_decode(answer, skip_special_tokens=True)[0]

    return {"response": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
