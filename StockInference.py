# Install required packages
!pip install datasets transformers peft accelerate torch wget
!pip install yfinance feedparser finnhub-python GoogleNews alpaca-trade-api nltk
# !pip install -q -U google-generativeai

import yfinance as yf
from GoogleNews import GoogleNews
import datetime
import re
import finnhub
import time
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display, Markdown
from google.colab import userdata
import transformers
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Set up Finnhub client
finnhub_client = finnhub.Client(api_key='your_finnhub_api_key')

# Set up GoogleNews client
googlenews = GoogleNews()

# Configure Generative AI with API key
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Define a function to format text as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Print available models that support content generation
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

# Define the model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

# Clone the repository and install llama
!git clone https://github.com/facebookresearch/llama.git
!cd llama && pip install -e .

# Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/llama-3-8b-bnb-4bit',
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
model = model.eval()

# Define a function to fetch and process data
def fetch_data(tickers):
    today = datetime.datetime.now()
    one_month_ago = today - datetime.timedelta(days=30)

    results = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')
        start_price = hist.iloc[0]['Close']
        end_price = hist.iloc[-1]['Close']
        increase_decrease = "increased" if end_price > start_price else "decreased"

        # Fetching news via Finnhub
        finnhub_news = finnhub_client.company_news(ticker, _from=one_month_ago.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))

        # Fetching news via GoogleNews
        googlenews.set_time_range(one_month_ago.strftime('%m/%d/%Y'), today.strftime('%m/%d/%Y'))
        googlenews.search(ticker)
        google_news_items = googlenews.results()

        news_prompt = ""
        for news in finnhub_news[:5]:
            news_prompt += f"[Finnhub Headline]: {news.get('headline', 'No headline available')}\n[Finnhub Summary]: {news.get('summary', 'No summary available')}\n\n"

        for news in google_news_items[:5]:
            news_prompt += f"[Google News Title]: {news['title']}\n[Google News Date]: {news['date']}\n[Google News Source]: {news['media']}\n\n"

        company_info = f"""
        [Company Introduction]:

        {stock.info.get('longName', 'N/A')} is a leading entity in the {stock.info.get('industry', 'N/A')} sector. Incorporated and publicly traded since {stock.info.get('ipoYear', 'an unknown year')}, the company has established its reputation as one of the key players in the market. As of today, {stock.info.get('longName', 'N/A')} has a market capitalization of {stock.info.get('marketCap', 'N/A')} in USD, with {stock.info.get('sharesOutstanding', 'N/A')} shares outstanding. {stock.info.get('longName', 'N/A')} operates primarily in the {stock.info.get('country', 'N/A')}, trading under the ticker {ticker} on the {stock.info.get('exchange', 'N/A')}. As a dominant force in the {stock.info.get('industry', 'N/A')} space, the company continues to innovate and drive progress within the industry.

        From {one_month_ago.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}, {stock.info.get('longName', 'N/A')}'s stock price {increase_decrease} from {start_price} to {end_price}. Company news during this period are listed below:

        {news_prompt}
                Based on all the information before {today.strftime('%Y-%m-%d')}, let's first analyze the positive developments and potential concerns for {ticker}. Most factors should be inferred from company-related news:

        [Positive Developments]:
        1. ...
        2. ...

        [Potential Concerns]:
        1. ...
        2. ...

        [Prediction & Analysis]:
        Based on the current trends and market conditions, I predict that the stock price of {ticker} will [increase/decrease] today and [increase/decrease] in the upcoming week due to [reasons].
        """
        results.append(company_info)

    return results

# Define a function to generate forecast
def generate_forecast(tickers):
    SYSTEM_PROMPT = fetch_data(tickers)
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    YOUR_PROMPT = """
    Make your prediction based on the current trends and market conditions for today and the upcoming week. Say if it will increase or decrease and by how much.
    """
    prompts = B_INST + B_SYS + "\n".join(SYSTEM_PROMPT) + E_SYS + YOUR_PROMPT + E_INST
    return prompts

# Example tickers
tickers = ['AAPL']

# Generate and print forecast
for ticker in tickers:
    print(ticker)
    prompt = generate_forecast(ticker)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate forecast using base model
    base_output = base_model.generate(**inputs, max_length=4096, do_sample=True, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    base_answer = tokenizer.decode(base_output[0], skip_special_tokens=True)
    base_answer = re.sub(r'.*\[/INST\]\s*', '', base_answer, flags=re.DOTALL)
    print(base_answer)
    
    # Generate forecast using fine-tuned model
    output = model.generate(**inputs, max_length=4096, do_sample=True, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', answer, flags=re.DOTALL)
    print(answer)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

# Function to extract the first number from text
def extract_number_and_assign(text):
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    if numbers:
        return float(numbers[0])
    return None

# Example usage to extract number from the model outputs
EST_model = extract_number_and_assign(answer)
print("Extracted number (Fine-tuned model):", EST_model)
EST_base_model = extract_number_and_assign(base_answer)
print("Extracted number (Base model):", EST_base_model)
