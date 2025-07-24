import os
import zipfile
import shutil
import re
import unicodedata
import olefile
import zlib
import struct
from langchain_core.documents import Document

def remove_chinese_characters(s: str) -> str:
    return re.sub(r'[\u4e00-\u9fff]+', '', s)

def remove_control_characters(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

class HWPExtractor:
    FILE_HEADER_SECTION = "FileHeader"
    HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
    SECTION_NAME_LENGTH = len("Section")
    BODYTEXT_SECTION = "BodyText"
    HWP_TEXT_TAGS = [67]
    def __init__(self, filename):
        self._ole = olefile.OleFileIO(filename)
        self._dirs = self._ole.listdir()
        if not self.is_valid(): raise Exception("유효한 HWP 파일이 아닙니다.")
        self._compressed = self.is_compressed()
        self.text = self._get_text()
    def is_valid(self) -> bool:
        return [self.FILE_HEADER_SECTION] in self._dirs and [self.HWP_SUMMARY_SECTION] in self._dirs
    def is_compressed(self) -> bool:
        header = self._ole.openstream("FileHeader")
        header_data = header.read()
        return (header_data[36] & 1) == 1
    def get_body_sections(self) -> list[str]:
        m = [int(d[1][self.SECTION_NAME_LENGTH:]) for d in self._dirs if d[0] == self.BODYTEXT_SECTION]
        return ["BodyText/Section" + str(x) for x in sorted(m)]
    def get_text(self) -> str: return self.text
    def _get_text(self) -> str:
        sections = self.get_body_sections()
        text_parts = [self._get_text_from_section(section) for section in sections]
        return "\n".join(text_parts)
    def _get_text_from_section(self, section: str) -> str:
        bodytext = self._ole.openstream(section)
        data = bodytext.read()
        unpacked_data = zlib.decompress(data, -15) if self._compressed else data
        size = len(unpacked_data)
        i = 0
        text = ""
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff
            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                decoded_text = rec_data.decode('utf-16', errors='ignore')
                cleaned_text = remove_control_characters(remove_chinese_characters(decoded_text))
                text += cleaned_text
            i += 4 + rec_len
        return text

def load_hwp_text_with_extractor(file_path: str) -> list[Document]:
    extractor = HWPExtractor(file_path)
    full_text = extractor.get_text()
    return [Document(page_content=full_text, metadata={"source": os.path.basename(file_path)})]

def unzip_and_cleanup(directory: str):
    print("\n--- ZIP 파일 자동 압축 해제 시작 ---")
    zip_found = False
    for filename in list(os.listdir(directory)):
        if filename.lower().endswith('.zip'):
            zip_found = True
            zip_path = os.path.join(directory, filename)
            print(f"ZIP 파일 발견: '{filename}'. 압축을 해제합니다...")
            try:
                unzip_target_path = os.path.join(directory, os.path.splitext(filename)[0])
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                print(f" -> '{filename}' 파일 압축 해제 완료.")
                os.remove(zip_path)
                print(f" -> 원본 ZIP 파일 '{filename}' 삭제 완료.")
                if os.path.isdir(unzip_target_path):
                    print(f" -> 하위 폴더 '{os.path.basename(unzip_target_path)}'의 파일들을 상위로 이동합니다.")
                    for item in os.listdir(unzip_target_path):
                        shutil.move(os.path.join(unzip_target_path, item), os.path.join(directory, item))
                    os.rmdir(unzip_target_path)
                    print(f" -> 빈 폴더 '{os.path.basename(unzip_target_path)}' 삭제 완료.")
            except Exception as e:
                print(f"'{filename}' 처리 중 오류 발생: {e}")
    if not zip_found:
        print(" -> 처리할 ZIP 파일이 없습니다.")
    print("--- ZIP 파일 처리 완료 ---\n")
