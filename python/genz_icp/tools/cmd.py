import glob
import os
from pathlib import Path
from typing import Optional

import typer

from genz_icp.datasets import available_dataloaders, dataset_factory
from genz_icp.pipeline import OdometryPipeline


def guess_dataloader(data: Path, default_dataloader: str):
    if data.is_file():
        if data.suffix == ".pcap":
            return "ouster", data
    elif data.is_dir():
        if len(glob.glob(os.path.join(data, "*.pcap"))) > 0:
            return "ouster", data
    return default_dataloader, data


def name_callback(value: Optional[str]):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_available_dl_help = available_dataloaders()
if "generic" in _available_dl_help:
    _available_dl_help.remove("generic")

_docstring = f"""
:green_circle: GenZ-ICP pipeline runner\n
\b
[bold green]Examples: [/bold green]
$ genz_icp_pipeline --visualize <data-dir>
$ genz_icp_pipeline --dataloader kitti --sequence 07 <path-to-kitti-root>
$ genz_icp_pipeline --dataloader mulran <path-to-mulran-seq>
$ genz_icp_pipeline --dataloader apollo <path-to-apollo-root>
$ genz_icp_pipeline --dataloader ouster --ouster-meta <metadata.json> <path-to-ouster.pcap>
"""


@app.command(help=_docstring)
def genz_icp_pipeline(
    data: Path = typer.Argument(
        ..., help="The data directory or file used by the specified dataloader", show_default=False
    ),
    dataloader: Optional[str] = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported by GenZ-ICP",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, show_default=False, help="[Optional] Path to the config file"
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="[Optional] Open an online visualization of the GenZ-ICP pipeline",
        rich_help_panel="Additional Options",
    ),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a sequence",
        rich_help_panel="Additional Options",
    ),
    n_scans: int = typer.Option(
        -1,
        "--n-scans",
        "-n",
        show_default=False,
        help="[Optional] Number of scans to process, default is entire dataset",
        rich_help_panel="Additional Options",
    ),
    jump: int = typer.Option(
        0,
        "--jump",
        "-j",
        show_default=False,
        help="[Optional] Start processing from a given index",
        rich_help_panel="Additional Options",
    ),
    ouster_meta: Optional[Path] = typer.Option(
        None,
        "--ouster-meta",
        "-m",
        exists=True,
        show_default=False,
        help="[Optional] Ouster metadata json file path",
        rich_help_panel="Additional Options",
    ),
):
    if not dataloader:
        dataloader, data = guess_dataloader(data, default_dataloader="generic")

    if dataloader == "kitti" and not sequence:
        print('You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    dataset_kwargs = {}
    if dataloader == "kitti":
        dataset_kwargs["sequence"] = sequence
    if dataloader == "ouster" and ouster_meta is not None:
        dataset_kwargs["meta"] = str(ouster_meta)

    results = OdometryPipeline(
        dataset=dataset_factory(dataloader=dataloader, data_dir=data, **dataset_kwargs),
        config=config,
        visualize=visualize,
        n_scans=n_scans,
        jump=jump,
    ).run()
    results.print()


def run():
    app()


if __name__ == "__main__":
    run()


# Backwards compatible entrypoint

def main():
    run()
