import asyncio

from .server import main as _main


def cli() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    cli()


