import logging
import time
from pathlib import Path
import pandas as pd
from typing import Iterator
from uuid import uuid4
import json
from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import TextItem, ProvenanceItem
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
from mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.models.tesseract_ocr_model import TesseractOcrOptions

# Initialize logging
logging.basicConfig(level=logging.INFO)

IMAGE_RESOLUTION_SCALE = 1.0


class DoclingFileLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        load_dotenv()
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options.use_gpu = False
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        self.pipeline_options.generate_page_images = True
        # self.pipeline_options.generate_table_images = True
        # self.pipeline_options.generate_picture_images = True
        self.doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PPTX, InputFormat.PDF, InputFormat.IMAGE],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                ),
                InputFormat.PPTX: PowerpointFormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=MsPowerpointDocumentBackend,
                ),
                InputFormat.IMAGE: FormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseDocumentBackend,
                ),
            },
        )

    def lazy_load(self) -> list[LCDocument]:
        documents = []  # Initialize an empty list to store documents
        output_file_path = "docling_document_structure.txt"  # Define the output file path

        with open(output_file_path, 'w') as output_file:  # Open the file for writing
            for source in self._file_paths:
                try:
                    print(f"Processing {source}")
                    dl_doc = self.doc_converter.convert(source).document
                    
                    # Write the structure of the DoclingDocument to the output file
                    output_file.write(f"Docling Document structure for {source}:\n")
                    output_file.write(f"{dl_doc.__dict__}\n\n")  # Write the dictionary structure
                    print(f"Docling Document structure for {source} written to file.")

                    if dl_doc.tables:
                        for table_ix, table in enumerate(dl_doc.tables):
                            table_df: pd.DataFrame = table.export_to_dataframe()
                            print(f"## Table {table_ix}")
                            text = table_df.to_markdown()
                            documents.append(LCDocument(page_content=text))

                    else:
                        # Export text if there are no images
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
        try:
            return list(self.lazy_load())  # Will consume the generator
        except StopIteration:
            return []  # Handle StopIteration gracefully, return empty list
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
