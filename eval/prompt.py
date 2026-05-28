SYSTEM_PROMPT_TEMPLATE = (
    "You have access to the following functions. Use them if required:\n\n"
    "{tools_json}\n\n"
    'To call a function, respond with a JSON object of the format:\n'
    '{{"name": "function_name", "arguments": {{"key": "value"}}}}\n'
    "For multiple calls, respond with a JSON array of such objects."
)
