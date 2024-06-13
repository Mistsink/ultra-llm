from transformers import DataCollatorForLanguageModeling, GemmaTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer


# tokenizer = GemmaTokenizer.from_pretrained(
#             "meta-llama/Meta-Llama-3-8B",
#             cache_dir="/disk1/hy/models",
#             token="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",
#             padding_side="left",
#         )

tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            cache_dir="/disk1/hy/models",
            token="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",
            padding_side="left",
        )

model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            cache_dir="/disk1/hy/models",
            token="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",
        )

print(model.config.pad_token_id)
print(model)


# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# tokenized_datasets = [
#     "This is a sample text",
#     "This is another sample text",
#     "This is a third sample text",
#     "This is a fourth sample text",
#     "This is a fifth sample text",
#     "This is a sixth sample text",
# ]

# out = data_collator([tokenized_datasets[i] for i in range(5)])
# for key in out:
#     print(f"{key} shape: {out[key].shape}")
# # input_ids shape: torch.Size([5, 128])
# # attention_mask shape: torch.Size([5, 128])
# # labels shape: torch.Size([5, 128])