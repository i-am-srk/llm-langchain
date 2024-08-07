from langchain_community.document_loaders import OnlinePDFLoader
import pytesseract
import nltk

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'
# nltk.download('averaged_perceptron_tagger')

loader = OnlinePDFLoader('https://arxiv.org/pdf/2312.10997')
data = loader.load()
print(type(data[0]))