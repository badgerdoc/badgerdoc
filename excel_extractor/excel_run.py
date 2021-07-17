import click

from excel_extractor.excel_extractor import run_excel_job


@click.command()
@click.argument("file")
@click.argument("output")
def run(file: str, output: str):
    run_excel_job(file, output)


if __name__ == "__main__":
    run()
