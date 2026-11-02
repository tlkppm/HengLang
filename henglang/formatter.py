from __future__ import annotations

from dataclasses import fields, is_dataclass
import re

from .frontend import HengError, parse

TOKEN = re.compile(r'\s+|"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/|(?:\d+\.\d*(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)|(?:[^\W\d]\w*)|(?:->|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=)|[^\s]', re.UNICODE)


def semantic(value):
    if is_dataclass(value):
        return {field.name: semantic(getattr(value, field.name)) for field in fields(value) if field.name not in {"source", "line", "column", "end_line", "end_column", "sources", "source_files", "files"}}
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [semantic(item) for item in value]
    return value


def format_source(source: str, filename: str = "<memory>.heng") -> str:
    original = parse(source, filename)
    tokens = [match.group() for match in TOKEN.finditer(source) if not match.group().isspace()]
    output: list[str] = []
    line = ""
    indent = 0

    def flush() -> None:
        nonlocal line
        if line.strip():
            output.append("    " * indent + line.strip())
        line = ""

    previous = ""
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token.startswith("//") or token.startswith("/*"):
            if line:
                line = line.rstrip() + " "
            line += token
            flush()
        elif token == "{":
            line = line.rstrip() + " {"
            flush()
            indent += 1
        elif token == "}":
            flush()
            indent -= 1
            line = "}"
            if following == "else":
                line += " "
            elif following not in {";", ",", ")", "]", "."}:
                flush()
        elif token == ";":
            line = line.rstrip() + ";"
            flush()
        elif token == ",":
            line = line.rstrip() + ", "
        elif token == ":":
            line = line.rstrip() + ": "
        elif token == ".":
            line = line.rstrip() + "."
        elif token == "(":
            line = line.rstrip() + (" (" if previous in {"if", "while"} else "(")
        elif token == "[":
            line += "["
        elif token in {")", "]"}:
            line = line.rstrip() + token
        elif token in {"+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "->", "+=", "-=", "*=", "/="}:
            line = line.rstrip() + " " + token + " "
        elif token == "!":
            if line and not line.endswith((" ", "(", "[")):
                line += " "
            line += "!"
        else:
            if line and not line.endswith((" ", "(", "[", ".", "!")):
                line += " "
            line += token
        previous = token
    flush()
    result = "\n".join(output) + "\n"
    if semantic(original) != semantic(parse(result, filename)):
        raise HengError("格式化前后语法结构不一致，拒绝修改文件")
    return result
