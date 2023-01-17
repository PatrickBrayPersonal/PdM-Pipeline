"""
This is the entry point of your pipeline.

This is where you import the pipeline function from its module and resolve it.
"""
import fire

from pdm.pipeline import pipeline


def main(config: str):
    """
    Entry point of my pipeline.
    """
    pipeline(config)


if __name__ == "__main__":
    fire.Fire(main)
