import glob
import logging
import os
import pandas as pd
import pdb
import datetime
import pickle
import gzip


from zipfile import ZipFile
from bs4 import BeautifulSoup
from tqdm import tqdm
from settings import DATA_DIR
from src import utils


def main():
    utils.setup_logging(os.path.basename(__file__))
    logging.info(f"Start parsing.")
    files = get_downloaded_zip_files()
    for file in tqdm(files):
        extracted = extract(file)
        parse_and_store(file, extracted)
        remove(extracted)


def get_downloaded_zip_files(directory=os.path.join(DATA_DIR, 'raw')):
    """Function for collecting filenames of already downloaded raw files.
    @param directory: place where the files are stored
    @return: list of zip files
    """
    return glob.glob(os.path.join(directory, '*.zip'))


def extract(zip_file_name):
    """Function for extracting a single patent application record file and calling the parser for it.
    @param zip_file_name: the file
    """

    with ZipFile(zip_file_name, 'r') as zip:
        files_extracted = zip.namelist()
        try:
            assert len(files_extracted) == 1, f"{zip_file_name} contains more than one file: "
        except:
            logging.info(f"{zip_file_name} contains more than one file: {files_extracted}")
        zip.extract(files_extracted[0])
        return files_extracted[0]


def parse_and_store(file, extracted_file_name):
    abstract_dict, title_dict, description_dict, claims_dict, countrydict, applicationdatedict, publicationdatedict = parse_xml(
        extracted_file_name)
    df1 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Application Date"])
    df2 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Publication Date"])
    df3 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Assignee Country"])
    df = pd.concat([df1, df2, df3], axis=1)

    pdb.set_trace()

    # save pickle
    pickle_record(file, abstract_dict, title_dict, description_dict, claims_dict, df)


def pickle_record(cname, abstract_dict, title_dict, descr_dict, claims_dict, df):                # TODO: Add saving text and claims dict
    """Function for saving parsed data"""

    """parse file names"""
    dir_end = cname.rfind("/") + 1
    gzpickle_filename_abstract = cname[:dir_end] + "abstracts/abstracts_" + cname[dir_end:-4] + ".pkl.gz"
    gzpickle_filename_title = cname[:dir_end] + "titles/titles_" + cname[dir_end:-4] + ".pkl.gz"
    gzpickle_filename_description = cname[:dir_end] + "descriptions/descriptions_" + cname[dir_end:-4] + ".pkl.gz"
    gzpickle_filename_claims = cname[:dir_end] + "claims/claims_" + cname[dir_end:-4] + ".pkl.gz"
    gzpickle_filename_df = cname[:dir_end] + "df/df_" + cname[dir_end:-4] + ".pkl.gz"

    """pickle dicts and save"""
    for to_be_pickled, gzpickle_filename in [(abstract_dict, gzpickle_filename_abstract),
                                           (title_dict, gzpickle_filename_title),
                                           (descr_dict, gzpickle_filename_description),
                                           (claims_dict, gzpickle_filename_claims)]:
        if os.path.isfile(gzpickle_filename):
            datestring = datetime.date.strftime(datetime.datetime.now(),"-%d-%m-%Y-at-%H-%M-%S")
            newname = gzpickle_filename.split(".")[0] + datestring + "." + gzpickle_filename.split(".")[1] + "." + gzpickle_filename.split(".")[2]
            os.rename(gzpickle_filename, newname)
        #with open(pickle_filename, 'wb') as outputfile:
        #    pickle.dump(to_be_pickled, outputfile, pickle.HIGHEST_PROTOCOL)
        with gzip.GzipFile(gzpickle_filename, 'w') as outputfile:
            pickle.dump(to_be_pickled, outputfile, pickle.HIGHEST_PROTOCOL)
        #
        # To reload, do:
        #with gzip.GzipFile(gzpickle_filename, 'r') as gzf:
        #    reloaded = pickle.load(gzf)

    """pickle pandas and save"""
    df.to_pickle(gzpickle_filename_df, compression="gzip")


def remove(file_name, directory=os.path.join(DATA_DIR, 'raw')):
    os.remove(file_name)
    os.system("mv {0:s} {1:s}/ > /dev/null 2>&1".format(file_name, directory))


def parse_xml(filename):
    """Function for parsing the xml of the extracted patent application record file and recording
            abstract
            title
            description
            claims
        by patent."""
    abstractdict = {}
    titledict = {}
    descriptiondict = {}
    claimsdict = {}
    countrydict = {}
    applicationdatedict = {}
    publicationdatedict = {}
    with open(filename, "r", encoding="utf8", errors='ignore') as rfile:
        xmlopen = False
        xmlblock = ""
        for line in rfile:
            if line[:6] == "<?xml ":
                if xmlopen:  # new xml block
                    applID, title, abstract, description, claims, adate, pdate, ctry = parse_single_xml_patent(xmlblock)
                    if applID is not None:
                        abstractdict[applID] = abstract
                        titledict[applID] = title
                        descriptiondict[applID] = description
                        claimsdict[applID] = claims
                        countrydict[applID] = ctry
                        applicationdatedict[applID] = adate
                        publicationdatedict[applID] = pdate
                    xmlblock = ""
                    xmlopen = False
                xmlopen = True
                xmlblock += line
            else:
                xmlblock += line
            # pfile =  open("ipg050419.xml",'r')
        applID, title, abstract, description, claims = parse_single_xml_patent(xmlblock)
        if applID is not None:
            abstractdict[applID] = abstract
            titledict[applID] = title
            descriptiondict[applID] = description
            claimsdict[applID] = claims
            countrydict[applID] = ctry
            applicationdatedict[applID] = adate
            publicationdatedict[applID] = pdate

    return abstractdict, titledict, descriptiondict, claimsdict, countrydict, applicationdatedict, publicationdatedict


def parse_single_xml_patent(xml_block):
    """Function for parsing the xml for a single patent and extracting
            abstract
            title
            description
            claims."""
    soup = BeautifulSoup(xml_block, "lxml")
    applID = xml_get_applID(soup)
    if applID is not None:
        abstract = xml_get_abstract(soup)
        title = xml_get_title(soup)
        description = xml_get_description(soup)
        claims = xml_get_claims(soup)
        # if np.random.random() < 0.002:
        #    pdb.set_trace()
    else:
        abstract = None
        title = None
        description = None
        claims = None
    return applID, title, abstract, description, claims


def xml_get_publication_date(soup):
    """Function for extracting publication date from xml soup"""
    try:
        date = soup.find("us-patent-application").find("publication-reference").find("date").text
    except:
        date = None
    return date


def xml_get_application_date(soup):
    """Function for extracting application date from xml soup"""
    try:
        date = soup.find("us-patent-application").find("application-reference").find("date").text
    except:
        date = None
    return date


def xml_get_country(soup):
    """Function for extracting assignee country from xml soup"""
    try:
        country = soup.find("us-patent-application").find("us-parties").find("us-applicant").find("address").find(
            "country").text
    except:
        country = None
    return country


def xml_get_abstract(soup):
    """Function for extracting abstract from xml soup"""
    try:
        abstract = soup.find("us-patent-application").find("abstract").text
    except:
        abstract = None
    return abstract


def xml_get_applID(soup):
    """Function for extracting application ID from xml soup"""
    try:
        applID = soup.find("us-patent-application").find("document-id").find("doc-number").text
    except:
        applID = None
    return applID


def xml_get_title(soup):
    """Function for extracting title from xml soup"""
    try:
        title = soup.find("us-patent-application").find("invention-title").text
    except:
        title = None
    return title


def xml_get_description(soup):
    """Function for extracting description from xml soup"""
    try:
        d = soup.find("us-patent-application").find("description")
        # TODO: remove UTF8 codes (&#x2019; etc)                    # automatically
        # TODO: remove MATHS->math, table-> table, figref??         # Done
        # d = ???
        _ = [tag.extract() for tag in d(['table', 'maths'])]  # Cannot deal with '?in-line-formulae'
        d = d.text
    except:
        d = None
    return d


def xml_get_claims(soup):
    """Function for extracting claims from xml soup"""
    try:
        cl = soup.find("us-patent-application").find("claims").text
    except:
        cl = None
    return cl


if __name__ == '__main__':
    main()
