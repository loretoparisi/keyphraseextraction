# keyphrase_extraction_python
Keyphrase extraction in Python

## Disclaimer

This source code has beed adpted and ported to `Python3` from the original `Python2` source code available here 

[Intro to Automatic Keyphrase Extraction](https://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/)

## How to prepare the document
First, you need to retrie your document or paper Title, Abstract and Text. To convert your paper to text use a pdf converter like [PDFElement](https://pdf.wondershare.com/). To copy the text into a string use [this](https://onlinetexttools.com/json-stringify-text) tool.

## How to extract key phrases
As soon as you have filled the variables `title`, `abstract` and `text`, you can run the available algorithms:

- extract_candidate_chunks
- extract_candidate_words
- score_keyphrases_by_tfidf
- score_keyphrases_by_textrank
- extract_candidate_features
