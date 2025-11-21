from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from typing import List
import tempfile
import tarfile
import logging
import json
import re


DEFAULT_URL = "https://gcn.nasa.gov/circulars/archive.txt.tar.gz"



def header_regex_match(document: str, silent_errors: bool = False) -> dict:
    """
    Parses the header of a GCN circular

    Args:
        document (str): The content of the GCN circular
    
    Returns:
        dict: A dictionary containing the parsed header information
    """
    # metadata structure
    metadata = {
        "circularId": '',
        "subject": '',
        "createdOn": '',
        "submitter": '',
        # "eventId": '',
        "email": '',
    }

    # Regular expression pattern to match the header
    pattern = re.compile(r"""
        TITLE:\s*(.*?)\s*
        NUMBER:\s*(.*?)\s*
        SUBJECT:\s*(.*?)\s*
        DATE:\s*(.*?)\s*
        FROM:\s*(.*?)(?:\s*\n|$) # 忽略行尾的空白和换行符
    """, re.VERBOSE)
    match = pattern.search(document)

    # match check
    if not match or match.group(1) != 'GCN CIRCULAR':
        if silent_errors:
            logging.debug(f"Failed to parse document:\n{document}")
            return metadata
        else:
            raise Exception(f"Failed to parse document:\n{document}")

    # metadata structure
    metadata.update({
        "circularId": match.group(2),
        "subject": match.group(3),
        "createdOn": match.group(4),
        "submitter": match.group(5),
    })

    # event_match = re.match(r"^(.*?):", match.group(3))
    # if event_match:
    #     metadata.update({"eventId": event_match.group(1)})
    # else:
    #     logging.debug(f"Failed to parse eventId from subject: {match.group(3)}")

    email_match = re.search(r'<([^>]+)>', match.group(5))
    if email_match:
        metadata.update({"email": email_match.group(1)})
    else:
        logging.debug(f"Failed to parse email from submitter: {match.group(5)}")

    return metadata


def text_loader(file_path: str) -> Document:
    """
    Load a text file

    Args:
        file_path: The path to the text file

    Returns:
        A list of documents
    """
    loader = TextLoader(file_path, encoding="utf-8")
    doc = loader.load()[0]
    logging.debug("Parsing headers")
    doc.metadata.update(header_regex_match(doc.page_content))
    return doc


def directory_loader(folder_path: str, silent_errors: bool = False, show_progress: bool = True) -> List[Document]:
    """
    Load Text files from a directory

    Args:
        folder_path: The path to the directory
        silent_errors: Whether to silent errors
        show_progress: Whether to show progress

    Returns:
        A list of documents
    """
    logging.info(f"Loading gcn files from '{folder_path}'")
    loader = DirectoryLoader(folder_path, glob="**/*.txt", show_progress=show_progress, silent_errors=silent_errors, loader_cls=TextLoader)
    docs = loader.load()
    logging.debug(f"Loaded {len(docs)} documents")

    logging.debug("Parsing headers")
    for doc in docs:
        doc.metadata.update(header_regex_match(doc.page_content, silent_errors=silent_errors))
    return docs

def tarfile_loader(file_path: str, silent_errors: bool = False, show_progress: bool = True) -> List[Document]:
    """
    Load Text files from a tar file

    Args:
        file_path: The path to the tar file
        silent_errors: Whether to silent errors
        show_progress: Whether to show progress

    Returns:
        A list of documents
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(temp_dir, filter='tar')
            logging.info(f"Uzip '{file_path}' to temp dir '{temp_dir}'")

        return directory_loader(temp_dir, silent_errors=silent_errors, show_progress=show_progress)

def dict_list_loader(file_path: str):
    """
    Load a txt file with dict lines
    """
    dsets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dsets.append(d)
    return dsets