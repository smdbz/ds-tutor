from importlib.resources import files


def load_theory(name: str) -> str:
    return files("ds_tutor.notes").joinpath(f"{name}.md").read_text()
