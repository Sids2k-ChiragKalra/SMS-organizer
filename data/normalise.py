import re
import numpy as np
from nltk.stem import PorterStemmer

from data.features import hp_word_stemming

regex = {
	'decimal': r"\d*[.:,/\\]+\d+",
	'date': r'(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?'
            r'[a-z\s,.]*(?:\d{1,2}[-/th|st|nd|rd)\s,]*)+(?:\d{2,4})+',
	'number': r"(?<!\d)\d{4,25}(?!\d)",
	'special_chars': r"[\"/,:;_!<>()&^~`*#@+]+",
	'url':  r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+'
            r'[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
}


# removes lines
def remove_newlines(message):
	return message.replace('\n', ' ').replace('\r', ' ')


# trims all urls in sms text down to their domain names
def trim_urls(message):
	urls = re.findall(regex['url'], message)
	for url in urls:
		trimmed_url = url.split("//")[-1].split("/")[0].split('?')[0].replace('www.', '').split('.')[0]
		message = message.replace(url, trimmed_url)
	return message.strip()


# stem words to root meaning
def stem(message):
	message = trim_urls(message.lower())
	for pat in ['decimal', 'number', 'date', 'special_chars']:
		message = re.sub(regex[pat], ' ', message)  # space
	message = re.sub(r"[\-.?']+", '', message)  # no space

	if hp_word_stemming:
		ps = PorterStemmer()
		words = message.split()
		for w in words:
			if w is not None:
				stem_word = ps.stem(w)
				digit = sum([1 if re.match(r'\d', a) else 0 for a in stem_word]) > 0
				alpha = sum([1 if re.match(r'[A-Za-z]', a) else 0 for a in stem_word]) > 0
				if digit and alpha or len(stem_word) < 2:
					message = message.replace(w, '')
				else:
					message = message.replace(w, stem_word)
	return message


# changes DD MMM YYYY HH:MM time to [0,1]
def change_time(str_time):
	str_time = (str_time.split(' ')[3])  # get HH:MM
	series_time = int((60 * int(str_time.split(':')[0])) + int(str_time.split(':')[1]))
	series_time = (np.abs(np.abs(series_time - 240) - (60 * 12)))
	return series_time / 720


# returns normalised number of words in each message
def number_words(features):
	lengths = np.sum(features.astype(int), axis=1, keepdims=True)
	return lengths / 150


def has_dates(message):
	return int(bool(re.search(regex['date'], message)))


def has_numbers(message):
	return int(bool(re.search(regex['number'], message)))


def has_decimals(message):
	return int(bool(re.search(regex['decimal'], message)))


def has_urls(message):
	return int(bool(re.search(regex['url'], message)))
