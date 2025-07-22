import tiktoken
import openai

encoding = tiktoken.get_encoding("cl100k_base")  # TODO 查一下还有哪些编码方案

res = encoding.encode("tiktoken is great")
print(res)

# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# res = encoding.encode("tiktoken is great!")
# print(res)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    # 返回文本字符串中的Token数量, 用于计算API费用
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


'''
res = num_tokens_from_string("tiktoken is great!", "cl100k_base")
print(res)


res = encoding.decode([83, 1609, 5963, 374, 2294, 0])
print(res)

res = [encoding.decode_single_token_bytes(token) for token in [83, 1609, 5963, 374, 2294, 0]]
print(res)
# [b't', b'ik', b'token', b' is', b' great', b'!']
'''


def compare_encodings(example_string: str) -> None:
    print(f'\nExample string: "{example_string}"')
    # 打印编码结果, 解码结果, 直观的对比一下
    for encoding_name in ["gpt2", "p50k_base", "cl100k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print()
        print(f"{encoding_name}: {num_tokens} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")
        

# res = compare_encodings("antidisestablishmentarianism")
# print(res)

# res = compare_encodings("2 + 2 = 4")
# print(res)

# res = compare_encodings("お誕生日おめでとう")
# print(res)

