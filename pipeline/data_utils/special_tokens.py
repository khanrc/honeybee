SYSTEM = "The following is a conversation between a curious human and AI assistant."
SYSTEM_DETAIL = "The assistant gives helpful, detailed, and polite answers to the user's questions."
ORG_SYSTEM = SYSTEM + " " + SYSTEM_DETAIL
IMAGE = "<image>"
_MEDIA_TOKENS = {"image": [IMAGE]}  # Special tokens used in this codebase.
# Role pattern tokens
HUMAN = "Human: "
AI = "AI: "
