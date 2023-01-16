"""
This is the entry point of your pipeline.

This is where you import the pipeline function from its module and resolve it.
"""
# Sematic
from pdm.pipeline import pipeline
from sematic import SilentResolver
import fire


def main(config: str, silent: bool = False):
    """
    Entry point of my pipeline.
    """
    assert isinstance(silent, bool), "silent entry must be 'True' or 'False'"
    if silent:
        resolver = SilentResolver()
    else:
        resolver = None
    pipeline(config).set(name="PdM Development", tags=["DEBUG"]).resolve(resolver)


if __name__ == "__main__":
    fire.Fire(main)
