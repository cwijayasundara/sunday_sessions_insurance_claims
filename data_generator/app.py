from llama_index.core import SimpleDirectoryReader, Settings
import warnings
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from sample_data import sample_claim_form_data

warnings.filterwarnings('ignore')
_ = load_dotenv()

documents = SimpleDirectoryReader(input_files=["../docs/Claim_Form_Property.pdf"]).load_data()

document = documents[0]

Settings.llm = OpenAI(model="gpt-4o-mini", )

prompt_str = """System : You are an insurance data generation expert. 
Can you fill in this form {form} attribute by attribute with sensible data please? sample data can be found below: 
{sample_data}. 

DO NOT GENERATE ANY OTHER DATA AND JUST FILL IN THE FORM."""

prompt_tmpl = PromptTemplate(prompt_str)

prompt = prompt_tmpl.format_messages(form=document, sample_data=sample_claim_form_data)

resp = OpenAI().chat(prompt)

""" save the generated resp to the /data folder to a file called claim_form_data.txt """
with open("../data/claim_form_data.txt", "w") as f:
    f.write(str(resp))
    f.close()
