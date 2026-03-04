import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- 1. Load model + tokenizer ----
model_name = "/mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/granite-4.0-8b-base-prerelease-killington-final"  # change if needed

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ---- 2. Define new special token ----
new_special_token = "<|end_of_turn|>"

special_tokens_dict = {
    "additional_special_tokens": [new_special_token]
}

# ---- 3. Add token to tokenizer ----
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_tokens} new token(s)")

# ---- 4. Resize model embeddings ----
if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))

# Optional: initialize new token embedding more intelligently
# Example: copy EOS embedding
with torch.no_grad():
    if num_added_tokens > 0:
        new_token_id = tokenizer.convert_tokens_to_ids(new_special_token)
        eos_token_id = tokenizer.eos_token_id
        
        if eos_token_id is not None:
            model.get_input_embeddings().weight[new_token_id] = \
                model.get_input_embeddings().weight[eos_token_id]

# ---- 5. Save updated model + tokenizer ----
save_path = "/mnt/vast/proj/checkpoints/granite-4-models-carina/ckpts/granite-4.0-8b-base-prerelease-killington-final-eoturn"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved to {save_path}")
