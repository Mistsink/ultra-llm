from transformers import DataCollatorForLanguageModeling, GemmaTokenizer

tokenizer = GemmaTokenizer.from_pretrained(
            "google/gemma-2b",
            cache_dir="hf_models",
            token="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",
            padding_side="left",
        )


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

tokenized_datasets = [
    "This is a sample text",
    "This is another sample text",
    "This is a third sample text",
    "This is a fourth sample text",
    "This is a fifth sample text",
    "This is a sixth sample text",
]

out = data_collator([tokenized_datasets[i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
# input_ids shape: torch.Size([5, 128])
# attention_mask shape: torch.Size([5, 128])
# labels shape: torch.Size([5, 128])