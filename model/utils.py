import torch
from torch import nn

# Write a utility function to generate a sequence of tokens autoregressively given a model and a starting prompt
def generate_sequence(model : nn.Module, start_prompt, max_length, device):
    model.eval()
    generated = start_prompt
    input_ids = torch.tensor(start_prompt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)   # [batch_size, sequence_length, vocab_size]

            # Get the logits for the next token in the sequence (the last token in the output sequence)
            next_token_logits = outputs[:, -1, :]   # [batch_size, vocab_size]

            next_token = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
            # Append the predicted next token to the generated sequence and update the input_ids for the next iteration
            generated.append(next_token.item())

            # Update the input_ids for the next iteration by concatenating the new token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return generated


# Utility function to generate sequence susing temperature scaling and top-k sampling
def generate_sequence_with_temperature_and_top_k(model: nn.Module, start_prompt, max_length, device, temperature=1.0, top_k=0):
    model.eval()
    generated = start_prompt
    input_ids = torch.tensor(start_prompt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):

            # Create the causal attention mask for the current sequence length if needed
            seq_len = input_ids.size(1)
            causal_attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)  # [1, seq_len, seq_len]

            outputs = model(input_ids, causal_attn_mask=causal_attn_mask)   # [batch_size, sequence_length, vocab_size]

            # Get the logits for the next token in the sequence (the last token in the output sequence)
            next_token_logits = outputs[:, -1, :]   # [batch_size, vocab_size]

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature  # [batch_size, vocab_size]

            # Apply top-k sampling if top_k > 0
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                # Set all logits to float('-inf')
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                # Scatter the top-k values back into the logits tensor using the indices of the top-k elements
                next_token_logits.scatter_(1, top_k_indices, top_k_values)

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)  # [batch_size, 1]

            # Append the predicted next token to the generated sequence and update the input_ids for the next iteration
            generated.append(next_token.item())

            # Update the input_ids for the next iteration by concatenating the new token
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return generated