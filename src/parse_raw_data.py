import glob
import logging
import os
from typing import List

import pandas as pd

from zipfile import ZipFile
from bs4 import BeautifulSoup
from parsing_helpers import get_application_id, get_abstract, get_title, get_description, get_claims, \
    get_publication_date, get_application_date, get_country
from tqdm import tqdm
from settings import DATA_DIR
import utils


def main():
    utils.setup_logging(os.path.basename(__file__))
    logging.info(f"Start parsing.")
    files = get_downloaded_zip_files()
    for file_name in tqdm(files):
        lines = extract(file_name)
        patents = parse_xml(lines, file_name)
        df = pd.DataFrame.from_records(patents)
        file_name_without_path = file_name.split(os.sep)[-1]
        df.to_pickle(os.path.join(DATA_DIR, 'processed', file_name_without_path + '_patents.pkl.gz'), compression="gzip")


def get_downloaded_zip_files(directory=os.path.join(DATA_DIR, 'raw')):
    """Function for collecting filenames of already downloaded raw files.
    @param directory: place where the files are stored
    @return: list of zip files
    """
    return glob.glob(os.path.join(directory, '*.zip'))


def extract(zip_file_path: str) -> List[bytes]:
    """Function for extracting a single patent application record file and calling the parser for it.
    @param zip_file_path: the file
    """

    with ZipFile(zip_file_path, 'r') as zip_file:
        files_extracted = zip_file.namelist()
        try:
            assert len(files_extracted) == 1, f"{zip_file_path} contains more than one file: "
        except:
            logging.info(f"{zip_file_path} contains more than one file: {files_extracted}")
        with zip_file.open(files_extracted[0], 'r') as f:
            return f.readlines()


def parse_xml(content, file_name):
    """Function for parsing the xml of the extracted patent application record file and recording
            abstract
            title
            description
            claims
        by patent."""

    patents = []

    xml_block = ''
    for line in iter(content):
        line_str = line.decode()
        if line_str.startswith('<?xml') and xml_block != "":  # this means we are done with one xml block/patent

            patents.append(parse_single_patent(xml_block, file_name))
            xml_block = ""

        else:
            xml_block += line_str

    patents.append(parse_single_patent(xml_block, file_name))
    
    patents = [p for p in patents if p['application_id'] is not None]
    
    return patents


def parse_single_patent(xml_block: str, file_name: str):
    """Function for parsing the xml for a single patent and extracting
            abstract
            title
            description
            claims."""
    patent = {}
    soup = BeautifulSoup(xml_block, "lxml")

    patent['application_id'] = get_application_id(soup, file_name)
    if patent['application_id'] is not None:
        patent['abstract'] = get_abstract(soup, file_name, patent['application_id'])
        patent['title'] = get_title(soup, file_name, patent['application_id'])
        patent['description'] = get_description(soup, file_name, patent['application_id'])
        patent['claims'] = get_claims(soup, file_name, patent['application_id'])
        patent['publication_date'] = get_publication_date(soup, file_name, patent['application_id'])
        patent['application_date'] = get_application_date(soup, file_name, patent['application_id'])
        patent['country'] = get_country(soup, file_name, patent['application_id'])

    return patent


if __name__ == '__main__':
    main()
