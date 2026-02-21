import re

INDENT_SIZE = 4

CONTROL_DEDENT = (
    "else:",
    "elif ",
    "except:",
    "finally:",
)

def get_prompt_indent(prompt: str) -> int:
    lines = prompt.split("\n")
    if not lines:
        return 0
    last_line = lines[-1]
    return len(last_line) - len(last_line.lstrip(" "))

def clean_trailing_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)

def auto_indent_fix(prompt: str, completion: str) -> str:
    base_indent = get_prompt_indent(prompt)
    indent_level = base_indent
    lines = completion.split("\n")
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            fixed_lines.append("")
            continue

        if stripped.startswith(CONTROL_DEDENT):
            indent_level = max(base_indent, indent_level - INDENT_SIZE)

        fixed_lines.append(" " * indent_level + stripped)

        if stripped.endswith(":"):
            indent_level += INDENT_SIZE

        indent_level = max(base_indent, indent_level)

    result = "\n".join(fixed_lines)
    return clean_trailing_whitespace(result)