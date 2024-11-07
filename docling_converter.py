from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    PowerpointFormatOption,
    FormatOption,
    
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions
import pandas as pd
from typing import Iterator
from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter
import json
import logging
import time
from pathlib import Path

## Default initialization still works as before:
# doc_converter = DocumentConverter()


# previous `PipelineOptions` is now `PdfPipelineOptions`

#...

## Custom options are now defined per format.


class DoclingFileLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        load_dotenv()
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.doc_converter = (
            DocumentConverter(  # all of the below is optional, has internal defaults.
                allowed_formats=[
                    InputFormat.PPTX,
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                ],  # whitelist formats, non-matching files are ignored.
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipeline_options, # pipeline options go here.
                        backend=PyPdfiumDocumentBackend # optional: pick an alternative backend
                    ),
                    InputFormat.PPTX: PowerpointFormatOption(
                        pipeline_cls=SimplePipeline,
                        backend=MsPowerpointDocumentBackend # default for office formats and HTML
                    ),
                    InputFormat.IMAGE: FormatOption(
                        pipeline_cls=StandardPdfPipeline, 
                        backend=DoclingParseDocumentBackend
                    ),
                },
            )
        )

    def lazy_load(self) -> list[LCDocument]:
        documents = []  # Initialize an empty list to store documents
        for source in self._file_paths:
            try:
                print(f"Processing {source}")
                dl_doc = self.doc_converter.convert(source).document
            
                if dl_doc.tables:
                    for table_ix, table in enumerate(dl_doc.tables):
                        table_df: pd.DataFrame = table.export_to_dataframe()
                        print(f"## Table {table_ix}")
                        text = table_df.to_markdown()
                        documents.append(LCDocument(page_content=text))
                else:
                    text = dl_doc.export_to_markdown()
                    documents.append(LCDocument(page_content=text))  # Append the document to the list
            except PermissionError as e:
                print(f"PermissionError: {e}")
                print("Ensure that the file is not open in another program and you have the necessary permissions.")
                continue  # Skip this file and continue with others
            except FileNotFoundError as e:
                print(f"FileNotFoundError: {e}")
                print("Ensure that the file exists at the specified path.")
                continue  # Skip this file and continue with others
            except Exception as e:
                print(f"Unexpected error processing {source}: {e}")
                continue  # Skip this file and continue with others
        return documents  # Return the list of documents


    def load(self):
        # Correct way to handle the generator and avoid RuntimeError
        try:
            return list(self.lazy_load())  # Will consume the generator
        except StopIteration:
            return []  # Handle StopIteration gracefully, return empty list
        except Exception as e:
            # Catch other possible exceptions (e.g., PermissionError, FileNotFoundError)
            print(f"An error occurred: {e}")
            return []