import os
import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma")
COLLECTION_NAME = "law-docs"


def upload_all_docs():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    docx_files = glob.glob(os.path.join(DOCS_DIR, "*.docx"))

    if not docx_files:
        print("docs 폴더에 .docx 파일이 없습니다.")
        return

    all_docs = []
    for filepath in docx_files:
        filename = os.path.basename(filepath)
        print(f"로딩 중: {filename}")
        loader = Docx2txtLoader(filepath)
        docs = loader.load_and_split(text_splitter=text_splitter)
        all_docs.extend(docs)

    print(f"\n총 {len(all_docs)}개 청크 생성 완료")
    print(f"Chroma에 저장 중... (persist_directory: {CHROMA_DIR})")

    Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print("저장 완료!")


if __name__ == "__main__":
    upload_all_docs()
