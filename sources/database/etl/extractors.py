import subprocess
import os


def download_data(
    data_source: str, destination_folder: str, file_name: str
) -> None:
    """
    Downloads data from a specified data source and saves it to a destination folder with a specified file name.

    Parameters:
        data_source (str): The URL or path of the data source.
        destination_folder (str): The path of the folder where the data will be saved.
        file_name (str): The name of the file to be saved.

    Returns:
        None
    """
    print(os.listdir("."))
    with open("./sources/database/etl/download.sh", "r") as f:
        s = f.read()
    with open("./sources/database/etl/download_.sh", "w") as f:
        f.write(
            s.format(
                data_source=data_source,
                persistent_data_path=destination_folder,
                file_name=file_name,
            )
        )
    subprocess.run(
        ["bash", "./sources/database/etl/download_.sh"],
        capture_output=True,
    )
    subprocess.run(["rm", "./sources/database/etl/download_.sh"])
