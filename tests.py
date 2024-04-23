import unittest


class TokenizerTestCase(unittest.TestCase):
    def test_tokenizer(self):
        import os
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["PRETRAINED"])
        messages = [
            {"role": "user", "content": "Hi, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you."},
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "My name is Mike."},
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": "My favorite color is blue."}
        ]

        token_list = []

        previous_prompt = ''
        for i in range(1, len(messages) + 1):
            if messages[i - 1]['role'] == 'system':
                continue
            agp_flag = messages[i - 1]['role'] == 'user'
            prompt = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=agp_flag)

            delta = prompt[len(previous_prompt):]
            previous_prompt = prompt
            token_list.extend(tokenizer.encode(delta, add_special_tokens=False))

            encoded_ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=agp_flag)
            self.assertEquals(encoded_ids, token_list)

        self.assertEquals(token_list, tokenizer.apply_chat_template(messages))


if __name__ == '__main__':
    unittest.main()
