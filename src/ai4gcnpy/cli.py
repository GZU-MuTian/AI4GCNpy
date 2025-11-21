import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def hello(
    name: str = typer.Argument(..., help="Name to input CSV"),
    formal: bool = False,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
):
    if formal:
        console.print(f"Good day, Ms. {name}.")
    else:
        console.print(f"Hello, {name}!")


@app.command()
def graph_build(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
):
    console.print(f"Building!")

if __name__ == "__main__":
    app()