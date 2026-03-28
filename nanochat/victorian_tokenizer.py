"""
Victorian BPE tokenizer wrapper — drop-in replacement for nanochat's RustBPETokenizer.

Wraps a HuggingFace tokenizer.json (byte-level BPE) and maps its special tokens
to nanochat's expected interface so that models trained with the Victorian tokenizer
(e.g. mr_chatterbox) work seamlessly with the nanochat inference stack.

Special token mapping:
  <|endoftext|>  → bos (document boundary)
  <|pad|>        → pad
  <human>        → user_start  (maps to nanochat's <|user_start|>)
  <victorian>    → assistant_start (maps to nanochat's <|assistant_start|>)
"""

import os
import copy
from pathlib import Path
from tokenizers import Tokenizer


class VictorianTokenizer:

    def __init__(self, tokenizer_path):
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self._tok.no_padding()
        self._tok.no_truncation()

    # ------------------------------------------------------------------
    # Core nanochat interface
    # ------------------------------------------------------------------

    def get_vocab_size(self):
        return self._tok.get_vocab_size()

    def get_bos_token_id(self):
        return self._tok.token_to_id("<|endoftext|>")

    def encode(self, texts, prepend=None, append=None, num_threads=4):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if isinstance(prepend, str):
            prepend = self.encode_special(prepend)
        if isinstance(append, str):
            append = self.encode_special(append)

        encodings = self._tok.encode_batch(texts, is_pretokenized=False)
        ids = [enc.ids for enc in encodings]

        if prepend is not None:
            ids = [[prepend] + seq for seq in ids]
        if append is not None:
            ids = [seq + [append] for seq in ids]

        return ids[0] if single else ids

    def decode(self, ids):
        return self._tok.decode(ids)

    # ------------------------------------------------------------------
    # Special token accessors
    # ------------------------------------------------------------------

    # Map from nanochat native tokens → Victorian equivalents
    _SPECIAL_MAP = {
        "<|assistant_start|>": "<victorian>",
        "<|assistant_end|>":   "<|endoftext|>",
        "<|user_start|>":      "<human>",
        "<|user_end|>":        "<|endoftext|>",
        "<|bos|>":             "<|endoftext|>",
        "<|eos|>":             "<|endoftext|>",
    }

    def encode_special(self, token):
        result = self._tok.token_to_id(token)
        if result is not None:
            return result
        mapped = self._SPECIAL_MAP.get(token)
        if mapped:
            return self._tok.token_to_id(mapped)
        return None

    def get_pad_token_id(self):
        return self._tok.token_to_id("<|pad|>")

    def get_user_start_id(self):
        return self._tok.token_to_id("<human>")

    def get_assistant_start_id(self):
        return self._tok.token_to_id("<victorian>")

    def id_to_token(self, id):
        return self._tok.id_to_token(id)

    # ------------------------------------------------------------------
    # Chat rendering (used by SFT / RL / inference)
    # ------------------------------------------------------------------

    def render_conversation(self, conversation, max_tokens=2048):
        human_id = self.get_user_start_id()
        victorian_id = self.get_assistant_start_id()
        bos_id = self.get_bos_token_id()

        # Handle both dict-with-"messages"-key and plain list forms
        if isinstance(conversation, dict):
            messages = conversation["messages"]
            # merge system message into first user message
            if messages[0]["role"] == "system":
                conversation = copy.deepcopy(conversation)
                messages = conversation["messages"]
                assert messages[1]["role"] == "user"
                messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
                messages = messages[1:]
        else:
            messages = conversation

        tokens = [bos_id]
        mask = [0]

        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            content_ids = self.encode(content)

            if role == "user":
                turn_tokens = [human_id] + content_ids
                turn_mask = [0] * len(turn_tokens)
            else:  # assistant
                turn_tokens = [victorian_id] + content_ids + [bos_id]
                turn_mask = [1] * len(turn_tokens)

            tokens.extend(turn_tokens)
            mask.extend(turn_mask)

            if len(tokens) >= max_tokens:
                tokens = tokens[:max_tokens]
                mask = mask[:max_tokens]
                break

        return tokens, mask

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

    # ------------------------------------------------------------------
    # Misc compatibility
    # ------------------------------------------------------------------

    def __call__(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def __repr__(self):
        return (
            f"VictorianTokenizer(vocab_size={self.vocab_size}, "
            f"bos={self.get_bos_token_id()}, "
            f"human={self.get_user_start_id()}, "
            f"victorian={self.get_assistant_start_id()})"
        )
