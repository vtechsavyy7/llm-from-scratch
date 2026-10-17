import torch
from torch import nn

# Write a utility function to generate a sequence of tokens autoregressively given a model and a starting prompt
def generate_sequence(model : nn.Module, start_prompt: list, max_new_tokens: int, device, use_kv_cache: bool = False):
    """Generate a sequence of tokens autoregressively given a model and a starting prompt.

    Args:
        model (nn.Module): The language model to use for generation.
        start_prompt (list): The initial sequence of token IDs to start generation from.
        max_new_tokens (int): The maximum number of new tokens to generate.
        device: The device to run the model on.
        use_kv_cache (bool, optional): Whether to use the key-value cache for faster decoding. Defaults to False.

    Returns:
        list: The generated sequence of token IDs, including the start prompt.
    """
    model.eval()

    generated_sequence = start_prompt   # List of generated token IDs, starting with the initial prompt
    input_query = torch.tensor(start_prompt, dtype=torch.long, device=device).unsqueeze(0) # Tensor of shape [1, sequence_length]

    with torch.no_grad():
        # Check if we are using the key-value cache for faster decoding
        if use_kv_cache:
            model.reset_kv_cache()   # Reset the key-value cache before starting generation
            # Run an initial forward pass to populate the key-value cache with the start prompt
            logits = model(input_query, attn_mask=None, use_kv_cache=True)  # [batch_size, sequence_length, vocab_size]

            for step in range(max_new_tokens):
                # Pick the best next token (greedy decoding)
                next_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]
                # Append the predicted next token to the generated sequence and update the input_ids for the next iteration
                generated_sequence.append(next_idx.item())
                input_query = torch.cat([input_query, next_idx], dim=1)
                # Run a forward pass with the updated input_ids to get the next logits
                if step + 1 < max_new_tokens:  # Skip the forward pass on the last step
                    logits = model(input_query, attn_mask=None, use_kv_cache=True)  # [batch_size, sequence_length, vocab_size]
        else: 
            for step in range(max_new_tokens):
                logits = model(input_query, attn_mask=None, use_kv_cache=False)  # [batch_size, sequence_length, vocab_size]
                next_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]
                generated_sequence.append(next_idx.item())
                input_query = torch.cat([input_query, next_idx], dim=1)

    return generated_sequence


# Utility function to generate sequence susing temperature scaling and top-k sampling
def generate_sequence_with_temperature_and_top_k(model: nn.Module, start_prompt: list, max_length: int, device, temperature=1.0, top_k=0):
    model.eval()
    generated = start_prompt
    input_ids = torch.tensor(start_prompt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):

            # Create the causal attention mask for the current sequence length if needed
            seq_len = input_ids.size(1)
            causal_attn_mask = torch.triu(
                                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                                    diagonal=1,
                                ).unsqueeze(0) # [1, seq_len, seq_len]

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