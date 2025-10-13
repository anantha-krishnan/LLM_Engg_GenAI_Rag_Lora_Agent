# ingest.py
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# --- CONFIGURATION ---
MDL_LIBRARY_PATH = "C:/Altair_Installs/2026_0_10_release/hwdesktop/hw/mdl/mdllib/Libs"
CSV_METADATA_PATH = "mdl_structure_metadata.csv"
DB_DIR = "./chroma_db_mdl"

import global_vars

@dataclass
class MdlComponent:
    file_path: str
    file_name: str
    description: str
    system_def_name: str
    attachment_candidates: List[str]
    # Metadata fields from CSV, with defaults
    model_type: str = "Unknown"
    side: str = "NA"
    sub_category_1: str = "Unknown"
    sub_category_2: str = "Unknown"

def parse_mdl_file(file_path: str) -> Dict[str, Any] | None:
    """Parses a single .mdl file for key information."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract description from comments
        description = ""
        for line in content.splitlines():
            if line.strip().startswith('//'):
                if 'Description:' in line:
                    description = line.split('Description:')[1].strip()
                    break # Found it, no need to continue
            elif line.strip(): # Stop if we hit a non-empty, non-comment line
                break
        
        #if not description:
        #    return None # Skip files without a standard description

        # Extract *DefineSystem info
        match = re.search(r"\*DefineSystem\(\s*(\w+)\s*,(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        system_name = "Unknown"
        attachments = []
        if match:
            system_name = match.group(1).strip()
            raw_args = match.group(2)
            attachments = [arg.strip().strip("'\"") for arg in raw_args.split(',') if arg.strip()]

        return {
            "description": description,
            "system_def_name": system_name,
            "attachment_candidates": attachments
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def build_mdl_vector_store(force_rebuild: bool = False):
    """Builds the Chroma vector store from .mdl files and CSV metadata."""
    if os.path.exists(DB_DIR) and not force_rebuild:
        print("Vector store already exists. Skipping build.")
        return

    print("--- Building new vector store ---")

    # 1. Load and process CSV metadata
    print(f"Loading metadata from {CSV_METADATA_PATH}...")
    metadata_df = pd.read_csv(CSV_METADATA_PATH)
    metadata_df.fillna("", inplace=True)
    # CRITICAL: Normalize paths to create a reliable lookup key
    metadata_df['full_path'] = (metadata_df['folder'].str.replace('\\', '/') + '/' + metadata_df['file_name']).str.lower()
    metadata_dict = metadata_df.set_index('full_path').to_dict(orient='index')
    print(f"Loaded {len(metadata_dict)} metadata entries.")

    # 2. Walk through MDL library and parse files
    all_components: List[MdlComponent] = []
    print(f"Scanning for .mdl files in {MDL_LIBRARY_PATH}...")
    for root, _, files in os.walk(MDL_LIBRARY_PATH):
        for file in files:
            if file.endswith(".mdl"):
                full_path = Path(root) / file
                # CRITICAL: Normalize path for matching
                normalized_path = str(full_path.as_posix()).lower()

                parsed_data = parse_mdl_file(str(full_path))
                if parsed_data:
                    # 3. Merge parsed data with CSV metadata
                    csv_meta = metadata_dict.get(normalized_path, {})
                    if not csv_meta:
                        print(f"  [!] Warning: No CSV metadata found for {file}")
                        continue # Skip files without metadata
                    component = MdlComponent(
                        file_path=str(full_path.as_posix()), # Store the clean path
                        file_name=file,
                        description=parsed_data['description'],
                        system_def_name=parsed_data['system_def_name'],
                        attachment_candidates=parsed_data['attachment_candidates'],
                        model_type=csv_meta.get('model_type', 'Unknown'),
                        side=csv_meta.get('side', 'NA'),
                        sub_category_1=csv_meta.get('sub_category_1', ''),
                        sub_category_2=csv_meta.get('sub_category_2', '')
                    )
                    all_components.append(component)

    print(f"Found and parsed {len(all_components)} valid .mdl files.")
    
    # 4. Create LangChain Documents
    documents = []
    for comp in all_components:
        page_content = (
            f"Component Type: {comp.model_type}, Side: {comp.side}, "
            f"Category: {comp.sub_category_1} {comp.sub_category_2}. "
            f"Description: {comp.description}"
        )
        metadata = {
            "file_name": comp.file_name,
            "file_path": comp.file_path,
            "model_type": comp.model_type,
            "side": comp.side,
            "category1": comp.sub_category_1,
            "category2": comp.sub_category_2,
            "system_def_name": comp.system_def_name,
            "attachment_candidates": ", ".join(comp.attachment_candidates),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        print("No documents were created. Aborting vector store build.")
        return

    # 5. Create and persist the vector store
    print(f"Creating embeddings and persisting {len(documents)} documents to {DB_DIR}...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=DB_DIR)
    vector_store.persist()
    print("--- Vector store build complete! ---")

if __name__ == "__main__":
    build_mdl_vector_store(force_rebuild=True)