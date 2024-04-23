from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    pretrained = "output"
    model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype="auto", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    prompt = "Who are you?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == '__main__':
    main()
