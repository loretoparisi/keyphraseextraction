# keyphrase_extraction_python
Keyphrase extraction in Python

## Updated
Added `Dockerfile` and simple module

## Disclaimer

This source code has beed adpted and ported to `Python3` from the original `Python2` source code available here 

[Intro to Automatic Keyphrase Extraction](https://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/)

## How to prepare the document
First, you need to retrie your document or paper Title, Abstract and Text. To convert your paper to text use a pdf converter like [PDFElement](https://pdf.wondershare.com/). To copy the text into a string use [this](https://onlinetexttools.com/json-stringify-text) tool.
We use [this](https://arxiv.org/abs/1901.04831) pre-print as example, **EXPLOITING SYNCHRONIZED LYRICS AND VOCAL FEATURES FOR MUSIC EMOTION DETECTION**

## How to extract key phrases
As soon as you have filled the variables `title`, `abstract` and `text`, you can run the available algorithms:

- extract_candidate_chunks
- extract_candidate_words
- score_keyphrases_by_tfidf
- score_keyphrases_by_textrank
- extract_candidate_features



```python
title = 'EXPLOITING SYNCHRONIZED LYRICS AND VOCAL FEATURES FOR MUSIC EMOTION DETECTION'
abstract = "One of the key points in music recommendation is authoring engaging playlists according to sentiment and emotions. While previous works were mostly based on audio for music discovery and playlists generation, we take advantage of our synchronized lyrics dataset to combine text representations and music features in a novel way; we therefore introduce the Synchronized Lyrics Emotion Dataset. Unlike other approaches that randomly exploited the audio samples and the whole text, our data is split according to the temporal information provided by the synchronization between lyrics and audio. This work shows a comparison between text-based and audio-based deep learning classiﬁcation models using different techniques from Natural Language Processing and Music Information Retrieval domains. From the experiments on audio we conclude that using vocals only, instead of the whole audio data improves the overall performances of the audio classiﬁer. In the lyrics experiments we exploit the state-ofthe-art word representations applied to the main Deep Learning architectures available in literature. In our benchmarks the results show how the Bilinear LSTM classiﬁer with Attention based on fastText word embedding performs better than the CNN applied on audio. "
text = "EXPLOITING SYNCHRONIZED LYRICS AND VOCAL FEATURES FOR \nMUSIC EMOTION DETECTION \n\nABSTRACT \nOne of the key points in music recommendation is authoring engaging playlists according to sentiment and emotions. While previous works were mostly based on audio for music discovery and playlists generation, we take advantage of our synchronized lyrics dataset to combine text representations and music features in a novel way; we therefore introduce the Synchronized Lyrics Emotion Dataset. Unlike other approaches that randomly exploited the audio samples and the whole text, our data is split according to the temporal information provided by the synchronization between lyrics and audio. This work shows a comparison between text-based and audio-based deep learning classiﬁcation models using different techniques from Natural Language Processing and Music Information Retrieval domains. From the experiments on audio we conclude that using vocals only, instead of the whole audio data improves the overall performances of the audio classiﬁer. In the lyrics experiments we exploit the state-ofthe-art word representations applied to the main Deep Learning architectures available in literature. In our benchmarks the results show how the Bilinear LSTM classiﬁer with Attention based on fastText word embedding performs better than the CNN applied on audio. \n\n1. INTRODUCTION \nMusic Emotion Recognition (MER) refers to the task of ﬁnding a relationship between music and human emotions [24,43]. Nowadays, this type of analysis is becoming more and more popular, music streaming providers are ﬁnding very helpful to present users with musical collections organized according to their feelings. The problem of Music Emotion Recognition was proposed for the ﬁrst time in the Music Information Retrieval (MIR) community in 2007, during the annual Music Information Research Evaluation eXchange (MIREX) [14]"
```

- extract_candidate_chunks
```python
set(extract_candidate_chunks(text))
{'100-dimensional word2vec',
 '] achieves relevant results',
 '] exploit contextual information',
 '] speech',
 '] techniques',
 'account',
 'acculturation',
 'acoustical properties',
 'adjectives',
 'advantage',
...
```

- extract_candidate_words
```python
set(extract_candidate_words(text))
{'100-dimensional',
 'abstract',
 'account',
 'acculturation',
 'achieves',
 'acoustic',
 'acoustical',
 'adjective',
 'adjectives',
 'adjectives/tags/labels',
 'advantage',
 'aforementioned',
 'agents',
 'aim',
 'allocation',
 ...
 ```
 
 - score_keyphrases_by_tfidf
 
 ```python
texts = [title, abstract, text]
corpus, corpus_tfidf, dictionary = score_keyphrases_by_tfidf(texts, candidates='chunks')
d = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc}
print(json.dumps(d,indent=4))
{
    "exploiting synchronized lyrics and vocal features for music emotion detection": 1.0,
    "advantage": 0.02818991144288086,
    "audio": 0.12685460149296385,
    "audio classi\ufb01er": 0.01409495572144043,
    "audio for music discovery": 0.01409495572144043,
    "audio samples": 0.01409495572144043,
    "audio-based deep learning classi\ufb01cation models": 0.01409495572144043,
    "benchmarks": 0.01409495572144043,
    "bilinear lstm classi\ufb01er with attention": 0.01409495572144043,
    "cnn": 0.02818991144288086,
    "comparison": 0.01409495572144043,
    "data": 0.01409495572144043,
    "different techniques from natural language processing": 0.01409495572144043,
    "emotions": 0.098664690050083,
    "experiments on audio": 0.01409495572144043,
    "fasttext word": 0.01409495572144043,
...
```
 
 - score_keyphrases_by_textrank
```python
 score_keyphrases_by_textrank(text)
 [('audio', 0.020772356010959545),
 ('audio features', 0.015940980040515616),
 ('music', 0.015056242947485416),
 ('lyrics', 0.014441885556177585),
 ('music emotion', 0.014074422805978502),
 ('emotion', 0.013092602664471589),
 ('music features', 0.01308292350877855),
 ('audio data', 0.012708279325535445),
 ('music information', 0.01175624019437015),
 ('emotions', 0.011338641233589263),
 ('features', 0.011109604070071685),
 ('music emotion classi ﬁcation', 0.010623139495342838),
 ('musical lyrics', 0.010621753673476389),
 ...
```

- extract_candidate_features
```python
candidates = extract_candidate_words(text)
candidates = candidates[0:5]
candidate_features = extract_candidate_features(candidates, text, abstract, title)
candidate_features = [{k[0]:k[1]} for k in sorted(candidate_features.items(), key=lambda item: item[1]['term_count'], reverse=True)]
print(json.dumps(candidate_features,indent=4))
[
    {
        "lyrics": {
            "term_count": 26,
            "term_length": 1,
            "max_word_length": 6,
            "spread": 0.9258158185840708,
            "lexical_cohesion": 0.0,
            "in_excerpt": 1,
            "in_title": 1,
            "abs_first_occurrence": 0.00165929203539823,
            "abs_last_occurrence": 0.9274751106194691
        }
    },
    {
        "features": {
            "term_count": 12,
            "term_length": 1,
            "max_word_length": 8,
            "spread": 0.9496681415929203,
            "lexical_cohesion": 0.0,
            "in_excerpt": 1,
            "in_title": 1,
            "abs_first_occurrence": 0.00283462389380531,
            "abs_last_occurrence": 0.9525027654867256
        }
    },
    {
        "synchronized": {
            "term_count": 11,
            "term_length": 1,
            "max_word_length": 12,
            "spread": 0.8889657079646018,
            "lexical_cohesion": 0.0,
            "in_excerpt": 1,
            "in_title": 1,
            "abs_first_occurrence": 0.0007605088495575221,
            "abs_last_occurrence": 0.8897262168141593
        }
    },
...
```

## How to build and run Docker

```
docker build -f Dockerfile -t keyphrase_extraction_python .
docker run --rm -it keyphrase_extraction_python bash
```
