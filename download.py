""" Script for downloading USPTO patent application data and extracting titles, abstracts, descriptions, claims, assignee countries, and 
application and publication dates.
"""

from zipfile import ZipFile
from bs4 import BeautifulSoup
import requests
import urllib
import os
import sys
import pickle
import datetime
import re
import pdb
import glob
import numpy as np
import gzip
from random import randint
from time import sleep

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
        country = soup.find("us-patent-application").find("us-parties").find("us-applicant").find("address").find("country").text
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
        #d = ???
        _ = [tag.extract() for tag in d(['table', 'maths'])]        # Cannot deal with '?in-line-formulae'
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

def parse_single_xml_patent(xml_block):
    """Function for parsing the xml for a single patent and extracting
            abstract
            title
            description
            claims."""
    soup = BeautifulSoup(xml_block, "xml")
    applID = xml_get_applID(soup)
    if applID is not None:
        abstract = xml_get_abstract(soup)
        title = xml_get_title(soup)
        description = xml_get_description(soup)
        claims = xml_get_claims(soup)
        #if np.random.random() < 0.002:
        #    pdb.set_trace()
    else:
        abstract = None
        title = None
        description = None
        claims = None
    return applID, title, abstract, description, claims

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
    rfile = open(filename, "r", encoding="utf8", errors='ignore')
    xmlopen = False
    xmlblock = ""
    for line in rfile:
        if line[:6] == "<?xml ":
            if xmlopen:             # new xml block
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
        #pfile =  open("ipg050419.xml",'r')
    applID, title, abstract, description, claims = parse_single_xml_patent(xmlblock)
    if applID is not None:
        abstractdict[applID] = abstract
        titledict[applID] = title
        descriptiondict[applID] = description
        claimsdict[applID] = claims
        countrydict[applID] = ctry
        applicationdatedict[applID] = adate
        publicationdatedict[applID] = pdate                    
    rfile.close()
    return abstractdict, titledict, descriptiondict, claimsdict, countrydict, applicationdatedict, publicationdatedict


def get_urls_year(year):
    """Function for obtaining all download urls for a given year"""
    urls = []
    names = []
    url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/" + year + "/"
    req = requests.get(url)
    soup = BeautifulSoup(req.text)
    for link in soup.find_all('a'):
        fname = link.get('href')
        full_url = url + fname
        if full_url.endswith('.zip'):
            urls.append(full_url)
            names.append(fname)
    return urls, names
    
    
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
    for to_be_pickled, gzpickle_filename in [(abstract_dict, gzpickle_filename_abstract), \
                                           (title_dict, gzpickle_filename_title), \
                                           (descr_dict, gzpickle_filename_description), \
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

def download_year(year, start=None, raw_file_directory="/mnt/usb4/patentapplications/raw_data/", extract_parse=True, reparse=False, 
                                                                work_directory="/mnt/usb4/patentapplications/"):
    """Function for downloading and parsing data by year"""
    
    os.chdir(work_directory)
    
    urls, names = get_urls_year(year) 

    if start is not None:
        nameidx = names.index(start)    
        names = names[nameidx:]
        urls = urls[nameidx:]

    for i, url in enumerate(urls):
        name = names[i]
        print("    Parsing item {0:2d}: {1:s}".format(i, name)
        
        was_downloaded = False
        
        if not os.path_exists("{1:s}/{0:s}".format(raw_file_directory,name)):
            print("        Item not present locally. Downloading.")
            print("        Fetching and parsing item {0:2d}: {1:s}".format(i, url))
            sleep(randint(4, 32))     # random wait
            urllib.request.urlretrieve(url, names[i])
            was_downloaded = True
        if extract_parse and (was_downloaded or reparse):
            extract_parse(name, raw_file_directory)
        
def extract_parse(name, raw_file_directory):        
    """Function for extracting a single patent application record file and calling the parser for it."""
    
    with ZipFile(name, 'r') as zip:
        files_extracted = zip.namelist()
        try:
            assert len(files_extracted)==1, "Zip {0:s} contains more than one file: ".format(names[i])
        except:
            print("Zip {0:s} contains more than one file: ".format(names[i]))
            print(files_extracted)
        zip.extract(files_extracted[0])

    abstract_dict, title_dict, description_dict, claims_dict, countrydict, applicationdatedict, publicationdatedict = parse_xml(files_extracted[0])
    df1 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Application Date"])
    df1 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Publication Date"])
    df3 = pd.DataFrame.from_dict(applicationdatedict, orient="index", columns=["Assignee Country"])
    df = pd.concat([df1, df2, df3], axis=1)
    
    pdb.set_trace()

    # rm extracted file
    os.remove(files_extracted[0])
    os.system("mv {0:s} {1:s}/ > /dev/null 2>&1".format(names[i], raw_file_directory))

    # save pickle
    pickle_record(name, abstract_dict, title_dict, description_dict, claims_dict, df)
        
def reread_files(raw_file_directory="/mnt/usb4/patentapplications/raw_data", work_directory="/mnt/usb4/patentapplications/raw_data/work/"):
    """Function for collecting filenames of already downloaded raw files and ralling extractor/parser again."""
    
    files = glob.glob("{0:s}/*.zip".format(raw_file_directory))
    names = [f.split("/")[-1] for f in files]
    os.chdir(work_directory)
    for name in names:
        extract_parse(name, raw_file_directory)

# main entry point
if __name__ == "__main__":
    
    """Define years and file to start with. list(range(2001, 2019)) and start=None is all files all years
        Years starting 2015 have version 4.4 and are consistent. 
        If possible we should stick to years 2015+."""
    years = [2020]
    start = None
    
    """Different example: several years, starting with a particular file in the first year:"""
    #years = list(range(2016, 2019))
    #start = "ipg160830.zip"
    
    """Different example: downloading off. Would still reread existing files"""
    years = []        # downloading off
    
    """Already downloaded files can be reread by calling reread_files()"""
    #reread_files()
    
    """Download, parse, and save"""
    for year in years:
        print("Commencing year {0:d}".format(year))
        download_year(str(year), start=start)
        
    """To just test the parse_file() function with a single file"""
    #files_extracted = ["ipg0200714.xml"]
    #abstract_dict, title_dict, description_dict, claims_dict = parse_xml(files_extracted[0])
    #pdb.set_trace()
