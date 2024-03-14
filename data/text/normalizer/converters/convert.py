import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import unicodedata

import os, sys
import re

from .Plain      import Plain
from .Punct      import Punct
from .Date       import Date
from .Letters    import Letters
from .Cardinal   import Cardinal
from .Verbatim   import Verbatim
from .Decimal    import Decimal
from .Measure    import Measure
from .Money      import Money
from .Ordinal    import Ordinal
from .Time       import Time
from .Electronic import Electronic
from .Digit      import Digit
from .Fraction   import Fraction
from .Telephone  import Telephone
from .Address    import Address
from .Roman    import Roman
from .Range    import Range


months = ['jan',
 'feb',
 'mar',
 'apr',
 'jun',
 'jul',
 'aug',
 'sep',
 'oct',
 'nov',
 'dec',
 'january',
 'february',
 'march',
 'april',
 'june',
 'july',
 'august',
 'september',
 'october',
 'november',
 'december']

labels = {
    "PLAIN": Plain(),
    "PUNCT": Punct(),
    "DATE": Date(),
    "LETTERS": Letters(),
    "CARDINAL": Cardinal(),
    "VERBATIM": Verbatim(),
    "DECIMAL": Decimal(),
    "MEASURE": Measure(),
    "MONEY": Money(),
    "ORDINAL": Ordinal(),
    "TIME": Time(),
    "ELECTRONIC": Electronic(),
    "DIGIT": Digit(),
    "FRACTION": Fraction(),
    "TELEPHONE": Telephone(),
    "ADDRESS": Address(),
    "ROMAN": Roman(),
    "RANGE": Range()
}

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

word_tokenize = TweetTokenizer().tokenize

def normalize_split(text):
    words = word_tokenize(text)
    chunks = split_given_size(words, 500)
    
    normalized_text = ""
    for words in chunks:
        sentence = TreebankWordDetokenizer().detokenize(words)
        normalized_text += normalizer.normalize(sentence) + " "
    
    return normalized_text.replace(" ' s", "'s")

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def is_oridinal(inputString):
    return inputString.endswith(("th", "nd", "st", "rd"))

def is_money(inputString):
    return inputString.startswith(('$', '€', '£', '¥'))

def is_time(inputString):
    return ":" in inputString

def is_cardinal(inputString):
    return "," in inputString or len(inputString) <= 3

def is_fraction(inputString):
    return "/" in inputString

def is_decimal(inputString):
    return "." in inputString

def is_range(inputString) : 
    return "-" in inputString

def is_url(inputString):
    return "//" in inputString or ".com" in inputString or ".html" in inputString

def has_month(inputString):
    return inputString.lower() in months or inputString == "May"

def normalize_single(text, prev_text = "", next_text = ""):
    if is_url(text):
        text = labels['ELECTRONIC'].convert(text).upper()
    elif has_numbers(text):
        if has_month(prev_text):
            prev_text = labels['DATE'].get_month(prev_text.lower())
            text = labels['DATE'].convert(prev_text + " " + text).replace(prev_text, "").strip()
        elif has_month(next_text):
            next_text = labels['DATE'].get_month(next_text.lower())
            text = labels['DATE'].convert(text + " " + next_text).replace(next_text, "").strip()
        elif is_oridinal(text):
            text = labels['ORDINAL'].convert(text)
        elif is_time(text):
            text = labels['TIME'].convert(text)
        elif is_money(text):
            text = labels['MONEY'].convert(text)
        elif is_fraction(text):
            text = labels['FRACTION'].convert(text)
        elif is_decimal(text):
            text = labels['DECIMAL'].convert(text)
        elif is_cardinal(text):
            text = labels['CARDINAL'].convert(text)
        elif is_range(text):
            text = labels['RANGE'].convert(text)
        else:
            text = labels['DATE'].convert(text)
        
        if has_numbers(text):
            text = labels['CARDINAL'].convert(text)
    elif text == "#" and has_numbers(next_text):
        text = "number"

    return text.replace("$", "")

def normalize_text(text):
    text = remove_accents(text).replace('–', ' to ').replace('-', ' - ').replace(":p", ": p").replace(":P", ": P").replace(":d", ": d").replace(":D", ": D")
    words = word_tokenize(text)

    df = pd.DataFrame(words, columns=['before'])

    df['after'] = df['before']
    
    df['previous'] = df.before.shift(1).fillna('') + "|" + df.before + "|" + df.before.shift(-1).fillna('')
    
    df['after'] = df['previous'].apply(lambda m: normalize_single(m.split('|')[1], m.split('|')[0], m.split('|')[2]))
    
    return TreebankWordDetokenizer().detokenize(df['after'].tolist()).replace("’ s", "'s").replace(" 's", "'s")

if __name__ == '__main__' : 
    text = 'hello (23 Jan 2020, 12:10 AM)'
    out = normalize_text(text)
    print(out)
