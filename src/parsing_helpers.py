import logging

from bs4 import BeautifulSoup


def get_application_id(soup: BeautifulSoup, file_name: str):
    """Function for extracting application ID from xml soup"""
    try:
        application_id = soup.find("us-patent-application").find("document-id").find("doc-number").text
    except Exception as e:
        application_id = None
        logging.error(f'Could not extract application id from {file_name}: {e}.')
    return application_id


def get_abstract(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting abstract from xml soup"""
    try:
        abstract = soup.find("us-patent-application").find("abstract").text
    except Exception as e:
        abstract = None
        logging.error(f'Could not extract abstract from {application_id} in {file_name}: {e}.')
    return abstract


def get_title(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting title from xml soup"""
    try:
        title = soup.find("us-patent-application").find("invention-title").text
    except Exception as e:
        title = None
        logging.error(f'Could not extract title from {application_id} in {file_name}: {e}.')
    return title


def get_description(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting description from xml soup"""
    try:
        d = soup.find("us-patent-application").find("description")
        # TODO: remove UTF8 codes (&#x2019; etc)                    # automatically
        # TODO: remove MATHS->math, table-> table, figref??         # Done
        if d is None:
            d = soup.find("description")
        if d is None:
            assert False, "No description present"
        # d = ???
        _ = [tag.extract() for tag in d(['table', 'maths'])]  # Cannot deal with '?in-line-formulae'
        d = d.text
    except Exception as e:
        d = None
        logging.error(f'Could not extract description from {application_id} in {file_name}: {e}.')
    return d


def get_claims(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting claims from xml soup"""
    try:
        cl = soup.find("claims").text
    except Exception as e:
        cl = None
        logging.error(f'Could not extract claims from {application_id} in {file_name}: {e}.')
    return cl


def get_publication_date(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting publication date from xml soup"""
    try:
        date = soup.find("us-patent-application").find("publication-reference").find("date").text
    except Exception as e:
        date = None
        logging.error(f'Could not extract publication date from {application_id} in {file_name}: {e}.')
    return date


def get_application_date(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting application date from xml soup"""
    try:
        date = soup.find("us-patent-application").find("application-reference").find("date").text
    except Exception as e:
        date = None
        logging.error(f'Could not extract application date from {application_id} in {file_name}: {e}.')
    return date


def get_country(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting assignee country from xml soup"""
    try:
        country = soup.find("us-patent-application").find("us-parties").find("us-applicant").find("address").find(
            "country").text
    except Exception as e:
        country = None
        logging.error(f'Could not extract country from {application_id} in {file_name}: {e}.')
    return country

def get_state(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting assignee (US) state from xml soup"""
    try:
        state = soup.find("us-patent-application").find("us-parties").find("us-applicant").find("address").find(
            "state").text
    except Exception as e:
        state = None
        logging.error(f'Could not extract state from {application_id} in {file_name}: {e}.')
    return state

def get_city(soup: BeautifulSoup, file_name: str, application_id: str):
    """Function for extracting assignee city from xml soup"""
    try:
        city = soup.find("us-patent-application").find("us-parties").find("us-applicant").find("address").find(
            "city").text
    except Exception as e:
        city = None
        logging.error(f'Could not extract city from {application_id} in {file_name}: {e}.')
    return city

