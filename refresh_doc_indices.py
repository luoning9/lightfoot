import openai
from llama_hub.file.cjk_pdf.base import CJKPDFReader
from llama_index import SimpleDirectoryReader, load_index_from_storage
import logging
import os
import myutils

global_conf = myutils.global_light.config
INDEX_ID = global_conf.get('default', 'INDEX_ID')

assert os.getenv("OPENAI_API_KEY") is not None, "please set openai key!"
assert os.getenv("all_proxy") is not None, "please set proxy!"
assert INDEX_ID, "no index id set!"

openai.api_key = os.getenv("OPENAI_API_KEY")
# ======= end of init ===============

storage_context = myutils.get_mongo_storage()

index = load_index_from_storage(storage_context, index_id=INDEX_ID)

logging.getLogger("pdfminer").setLevel(logging.INFO)
extractors = {'.pdf': CJKPDFReader(concat_pages=True)}
documents = SimpleDirectoryReader('examples/paul_graham_essay/data',
                                  file_extractor=extractors,
                                  filename_as_id=True).load_data()
for doc in documents:
    #    logging.debug("loaded doc %s", doc.text[:40]+"..." if len(doc.text)>43 else doc.text )
    doc_id = doc.get_doc_id()
    logging.debug(f"doc {doc_id} exists? {index.docstore.get_document_hash(doc_id)}")
updated = index.refresh_ref_docs(documents)
print(f"update status {updated}")
storage_context.persist()
