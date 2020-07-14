import argparse
import os
import logging
import requests

from bs4 import BeautifulSoup
from random import randint
from time import sleep

from settings import DATA_DIR
from src import utils
from urllib.parse import urljoin

from src.utils import get_headers

uspto_base_url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-y",
                        "--years",
                        help="list of years in [2001, 2019]",
                        type=lambda s: [int(item) for item in s.split(",")],
                        required=True)
    args = parser.parse_args()

    utils.setup_logging(os.path.basename(__file__))

    for year in args.years:
        logging.info(f"Start downloading year {year}.")
        download_year(year)
    logging.info("DONE")


def download_year(year):
    """Function for downloading data by year"""

    zip_files = get_zips_year(year)
    total = len(zip_files)
    for i, zip_file in enumerate(zip_files):
        logging.info(f"\tCheck item {i}/{total}: {zip_file}")

        out_path = os.path.join(DATA_DIR, 'raw', zip_file)

        if not os.path.exists(os.path.join(out_path)):
            url = f"{uspto_base_url}{year}/{zip_file}"
            logging.info(f"\tItem not present locally. Downloading.")
            logging.info(f"\tFetching item {i}: {url}.")

            response = requests.get(url, headers=get_headers())
            with open(out_path, 'wb') as f:
                f.write(response.content)

            sleep(randint(4, 32))  # random wait
        else:
            logging.info(f"\tSkip {zip_file} because already present.")


def get_zips_year(year):
    """
    Get all zip files for the given year
    @param year:
    @return:
    """
    url = urljoin(uspto_base_url, str(year))
    req = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(req.text, "html.parser")
    return [link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.zip')]


if __name__ == "__main__":
    main()
