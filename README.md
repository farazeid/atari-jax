# Getting Started

Install dependencies:
```zsh
uv sync  # explained below
```

Run the code:
```zsh
uv run cartpole.py
```

Start `mlflow` server locally:
```zsh
mlflow server
```


# Tools

## Astral uv
Package management to interact with instead of `pip`, `venv`, `conda`, etc.
Works nicely with `Docker` as well.

Creating a `uv`-centred codebase (packages listed in `pyproject.toml` instead of `requirements.txt`):
```zsh
uv init
uv add gym
uv remove gym
```

Picking-up an existing `uv` codebase that provides `pyproject.toml` and `uv.lock`:
```zsh
uv sync
```

Using a non-`uv` codebase with `uv`:
```zsh
uv init
uv add -r requirements.txt
```



## Astral ruff
Formatter and linter. (Translation of existing formatters and linters under one package, implemented in Rust.) Rulebook [link](https://docs.astral.sh/ruff/rules/) for understanding individual rules (E101) and rule sets (E).

Install as a global tool with `uv` (global tools and local packages are separated in the `pyproject.toml`):
```zsh
uv tool install ruff
```

VS Code extension [link](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

My additions to a `pyproject.toml`:
```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "N",   # pep8-naming
    "I",   # isort
    "B",   # flake8-bugbear
    "S",   # flake8-bandit
    "UP",  # pyupgrade
    "ANN", # flake8-annotations
    "SIM", # flake8-simplify
    "C90", # mccabe cyclomatic complexity
]
```



## Astral ty
Type-checker and language server.

VS Code extension [link](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty) provides live type annotation hints for untyped variables.

For example:
```python
def fn(x: int) -> str:
    ...

a = fn()
```
is rendered as, with the added type hint greyed-out:
```python
def fn(x: int) -> str:
    ...

a: str = fn()
```
