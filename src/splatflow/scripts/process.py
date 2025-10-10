import typer
from splatflow.process import hloc


def main():
    typer.run(hloc.run_hloc)
