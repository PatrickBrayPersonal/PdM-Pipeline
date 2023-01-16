"""
This is the entry point of your pipeline.

This is where you import the pipeline function from its module and resolve it.
"""
# Sematic
from pdm.pipeline import pipeline
from sematic import SilentResolver


def main():
    """
    Entry point of my pipeline.
    """
    pipeline().set(name="PdM Development", tags=["DEBUG"]).resolve()


if __name__ == "__main__":
    main()

# TODO: Build your CLI here, google CLI packages
