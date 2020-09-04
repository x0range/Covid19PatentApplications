import glob
import logging
import os
import argparse
import datetime
from typing import List

import pandas as pd

from zipfile import ZipFile
from bs4 import BeautifulSoup
from parsing_helpers import get_application_id, get_abstract, get_title, get_description, get_claims, \
    get_publication_date, get_application_date, get_country, get_state, get_city, get_classification
from tqdm import tqdm
from settings import DATA_DIR
import utils

def main():
    
    """ Handle arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("-s",
                        "--start",
                        help="Earliest date for parsed patent applications (anything before is ignored)",
                        type=str,
                        required=False)
    parser.add_argument("-e",
                        "--end",
                        help="Latest date for parsed patent applications (anything after is ignored)",
                        type=str,
                        required=False)
    args = parser.parse_args()
    
    if args.start:
        start_date = args.start
        assert int(start_date) and len(start_date) in [4, 6, 8], "Malformed start argument"
        if len(start_date) == 4:
            start_date += "01"
        if len(start_date) == 6:
            start_date += "01"
    else:
        start_date = None

    if args.end:
        end_date = args.end
        assert int(end_date) and len(end_date) in [4, 6, 8], "Malformed end argument"
        if len(end_date) == 4:
            end_date += "1231"
        if len(end_date) == 6:
            datetime_end_date = (pd.to_datetime(end_date, format="%Y%m") + datetime.timedelta(days=35)).replace(day=1) \
                                - datetime.timedelta(days=1)
            end_date = datetime_end_date.strftime("%Y%m%d")
    else:
        end_date = None
    
    """ Handle logging"""
    utils.setup_logging(os.path.basename(__file__))
    logging.info(f"Start parsing.")
    
    """ Obtain filenames"""
    files = get_downloaded_zip_files()
    
    """ Parse and save"""
    for file_name in tqdm(files):
        lines = extract(file_name)
        patents = parse_xml(lines, file_name, start_date, end_date)
        df = pd.DataFrame.from_records(patents)
        if len(df) == 0:
            df = pd.DataFrame(columns=['application_id', 'abstract', 'title', 'description', 'claims', 
                                       'publication_date', 'application_date', 'country', 'state', 'city', 
                                       'CPC_codes'])
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


def parse_xml(content, file_name, start_date: str, end_date: str):
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

            patents.append(parse_single_patent(xml_block, file_name, start_date, end_date))
            xml_block = ""

        else:
            xml_block += line_str

    patents.append(parse_single_patent(xml_block, file_name, start_date, end_date))
    
    patents = [p for p in patents if p['application_id'] is not None]
    
    return patents


def parse_single_patent(xml_block: str, file_name: str, start_date: str, end_date: str):
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
        patent['state'] = get_state(soup, file_name, patent['application_id'])
        patent['city'] = get_city(soup, file_name, patent['application_id'])
        patent['CPC_codes'] = get_classification(soup, file_name, patent['application_id'])
        
        if end_date is not None and \
                pd.to_datetime(patent["application_date"], format="%Y%m%d") > pd.to_datetime(end_date):
            patent = {'application_id': None}
        if (start_date is not None) and (patent.get("application_date") is not None) and \
                pd.to_datetime(patent["application_date"], format="%Y%m%d") < pd.to_datetime(start_date):
            patent = {'application_id': None}
        
    return patent


if __name__ == '__main__':
    main()
