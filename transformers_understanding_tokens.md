# Lesson 2: Comparing Trained LLM Tokenizers

In this notebook of lesson 2, you will work with several tokenizers associated with different LLMs and explore how each tokenizer approaches tokenization differently. 

## Setup

We start with setting up the lab by installing the `transformers` library and ignoring the warnings. The requirements for this lab are already installed, so you don't need to uncomment the following cell.

<p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> If you'd like to access the requirements file: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>


```python
%pip install transformers>=4.46.1
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.0[0m[39;49m -> [0m[32;49m25.1.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.



```python
# Warning control
import warnings
warnings.filterwarnings('ignore')
```

## Tokenizing Text

In this section, you will tokenize the sentence "Hello World!" using the tokenizer of the [`bert-base-cased` model](https://huggingface.co/google-bert/bert-base-cased). 

Let's import the `Autotokenizer` class, define the sentence to tokenize, and instantiate the tokenizer.

<p style="background-color:#fff1d7; padding:15px; "> <b>FYI: </b> The transformers library has a set of Auto classes, like AutoConfig, AutoModel, and AutoTokenizer. The Auto classes are designed to automatically do the job for you.</p>


```python
from transformers import AutoTokenizer
```


```python
# define the sentence to tokenize
sentence = "Hello world!"
```


```python
# load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```


```python
from transformers import AutoTokenizer

# Specify your own cache directory to make it visible
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir="./bert_cache")

```


    tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/436k [00:00<?, ?B/s]



```python
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers/")
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        if "bert-base-cased" in root:
            print(os.path.join(root, file))
```

You'll now apply the tokenizer to the sentence. The tokeziner splits the sentence into tokens and returns the IDs of each token.


```python
# apply the tokenizer to the sentence and extract the token ids
token_ids = tokenizer(sentence).input_ids
```


```python
print(token_ids)
```

    [101, 8667, 1362, 106, 102]


To map each token ID to its corresponding token, you can use the `decode` method of the tokenizer.


```python
for id in token_ids:
    print(tokenizer.decode(id))
```

    [CLS]
    Hello
    world
    !
    [SEP]



```python
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers/")
for root, dirs, files in os.walk(cache_dir):
    for file in files:
        if "bert-base-cased" in root:
            print(os.path.join(root, file))
```


```python
import os

for root, dirs, files in os.walk("./bert_cache"):
    for file in files:
        print(os.path.join(root, file))

```

    ./bert_cache/models--bert-base-cased/blobs/2ba5de7675473164e07f3b3531748c9a6f113a2c
    ./bert_cache/models--bert-base-cased/blobs/107460496b431545e4f921afb3fd5486fd2ae79d
    ./bert_cache/models--bert-base-cased/blobs/2ea941cc79a6f3d7985ca6991ef4f67dad62af04
    ./bert_cache/models--bert-base-cased/blobs/1ab2a0d23e5a032b6dcd6a3d0976c2af4d2c27f8
    ./bert_cache/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer_config.json
    ./bert_cache/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/config.json
    ./bert_cache/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/vocab.txt
    ./bert_cache/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer.json
    ./bert_cache/models--bert-base-cased/refs/main
    ./bert_cache/models--bert-base-cased/.no_exist/cd5ef92a9fb2f889e972770a36d4ed042daf221e/added_tokens.json
    ./bert_cache/models--bert-base-cased/.no_exist/cd5ef92a9fb2f889e972770a36d4ed042daf221e/special_tokens_map.json
    ./bert_cache/.locks/models--bert-base-cased/2ba5de7675473164e07f3b3531748c9a6f113a2c.lock
    ./bert_cache/.locks/models--bert-base-cased/107460496b431545e4f921afb3fd5486fd2ae79d.lock
    ./bert_cache/.locks/models--bert-base-cased/2ea941cc79a6f3d7985ca6991ef4f67dad62af04.lock
    ./bert_cache/.locks/models--bert-base-cased/1ab2a0d23e5a032b6dcd6a3d0976c2af4d2c27f8.lock



```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "Hello world!"
tokens = tokenizer(text, return_tensors="pt")
ids = tokens['input_ids'][0]

# View token IDs and their corresponding tokens
for token_id in ids:
    print(f"{token_id.item()} --> {tokenizer.convert_ids_to_tokens([token_id.item()])[0]}")

```

    101 --> [CLS]
    8667 --> Hello
    1362 --> world
    106 --> !
    102 --> [SEP]



```python

```

## Visualizing Tokenization

In this section, you'll wrap the code of the previous section in the function `show_tokens`. The function takes in a text and the model name, and prints the vocabulary length of the tokenizer and a colored list of the tokens. 


```python
# A list of colors in RGB for representing the tokens
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )
```

Here's the text that you'll use to explore the different tokenization strategies of each model.


```python
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""
```

You'll now again use the tokenizer of `bert-base-cased` and compare its tokenization strategy to that of `Xenova/gpt-4`

**bert-base-cased**


```python
show_tokens(text, "bert-base-cased")
```

    Vocab length: 28996
    [0;30;48;2;102;194;165m[CLS][0m [0;30;48;2;252;141;98mEnglish[0m [0;30;48;2;141;160;203mand[0m [0;30;48;2;231;138;195mCA[0m [0;30;48;2;166;216;84m##PI[0m [0;30;48;2;255;217;47m##TA[0m [0;30;48;2;102;194;165m##L[0m [0;30;48;2;252;141;98m##I[0m [0;30;48;2;141;160;203m##Z[0m [0;30;48;2;231;138;195m##AT[0m [0;30;48;2;166;216;84m##ION[0m [0;30;48;2;255;217;47m[UNK][0m [0;30;48;2;102;194;165m[UNK][0m [0;30;48;2;252;141;98mshow[0m [0;30;48;2;141;160;203m_[0m [0;30;48;2;231;138;195mtoken[0m [0;30;48;2;166;216;84m##s[0m [0;30;48;2;255;217;47mF[0m [0;30;48;2;102;194;165m##als[0m [0;30;48;2;252;141;98m##e[0m [0;30;48;2;141;160;203mNone[0m [0;30;48;2;231;138;195mel[0m [0;30;48;2;166;216;84m##if[0m [0;30;48;2;255;217;47m=[0m [0;30;48;2;102;194;165m=[0m [0;30;48;2;252;141;98m>[0m [0;30;48;2;141;160;203m=[0m [0;30;48;2;231;138;195melse[0m [0;30;48;2;166;216;84m:[0m [0;30;48;2;255;217;47mtwo[0m [0;30;48;2;102;194;165mta[0m [0;30;48;2;252;141;98m##bs[0m [0;30;48;2;141;160;203m:[0m [0;30;48;2;231;138;195m"[0m [0;30;48;2;166;216;84m"[0m [0;30;48;2;255;217;47mThree[0m [0;30;48;2;102;194;165mta[0m [0;30;48;2;252;141;98m##bs[0m [0;30;48;2;141;160;203m:[0m [0;30;48;2;231;138;195m"[0m [0;30;48;2;166;216;84m"[0m [0;30;48;2;255;217;47m12[0m [0;30;48;2;102;194;165m.[0m [0;30;48;2;252;141;98m0[0m [0;30;48;2;141;160;203m*[0m [0;30;48;2;231;138;195m50[0m [0;30;48;2;166;216;84m=[0m [0;30;48;2;255;217;47m600[0m [0;30;48;2;102;194;165m[SEP][0m 

**Optional - bert-base-uncased**

You can also try the uncased version of the bert model, and compare the vocab length and tokenization strategy of the two bert versions.


```python
show_tokens(text, "bert-base-uncased")
```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]


    Vocab length: 30522
    [0;30;48;2;102;194;165m[CLS][0m [0;30;48;2;252;141;98menglish[0m [0;30;48;2;141;160;203mand[0m [0;30;48;2;231;138;195mcapital[0m [0;30;48;2;166;216;84m##ization[0m [0;30;48;2;255;217;47m[UNK][0m [0;30;48;2;102;194;165m[UNK][0m [0;30;48;2;252;141;98mshow[0m [0;30;48;2;141;160;203m_[0m [0;30;48;2;231;138;195mtoken[0m [0;30;48;2;166;216;84m##s[0m [0;30;48;2;255;217;47mfalse[0m [0;30;48;2;102;194;165mnone[0m [0;30;48;2;252;141;98meli[0m [0;30;48;2;141;160;203m##f[0m [0;30;48;2;231;138;195m=[0m [0;30;48;2;166;216;84m=[0m [0;30;48;2;255;217;47m>[0m [0;30;48;2;102;194;165m=[0m [0;30;48;2;252;141;98melse[0m [0;30;48;2;141;160;203m:[0m [0;30;48;2;231;138;195mtwo[0m [0;30;48;2;166;216;84mtab[0m [0;30;48;2;255;217;47m##s[0m [0;30;48;2;102;194;165m:[0m [0;30;48;2;252;141;98m"[0m [0;30;48;2;141;160;203m"[0m [0;30;48;2;231;138;195mthree[0m [0;30;48;2;166;216;84mtab[0m [0;30;48;2;255;217;47m##s[0m [0;30;48;2;102;194;165m:[0m [0;30;48;2;252;141;98m"[0m [0;30;48;2;141;160;203m"[0m [0;30;48;2;231;138;195m12[0m [0;30;48;2;166;216;84m.[0m [0;30;48;2;255;217;47m0[0m [0;30;48;2;102;194;165m*[0m [0;30;48;2;252;141;98m50[0m [0;30;48;2;141;160;203m=[0m [0;30;48;2;231;138;195m600[0m [0;30;48;2;166;216;84m[SEP][0m 

**GPT-4**


```python
show_tokens(text, "Xenova/gpt-4")
```


    tokenizer_config.json:   0%|          | 0.00/460 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/2.01M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/917k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/4.23M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/98.0 [00:00<?, ?B/s]


    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


    Vocab length: 100263
    [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mEnglish[0m [0;30;48;2;141;160;203m and[0m [0;30;48;2;231;138;195m CAPITAL[0m [0;30;48;2;166;216;84mIZATION[0m [0;30;48;2;255;217;47m
    [0m [0;30;48;2;102;194;165m�[0m [0;30;48;2;252;141;98m�[0m [0;30;48;2;141;160;203m�[0m [0;30;48;2;231;138;195m �[0m [0;30;48;2;166;216;84m�[0m [0;30;48;2;255;217;47m�[0m [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mshow[0m [0;30;48;2;141;160;203m_tokens[0m [0;30;48;2;231;138;195m False[0m [0;30;48;2;166;216;84m None[0m [0;30;48;2;255;217;47m elif[0m [0;30;48;2;102;194;165m ==[0m [0;30;48;2;252;141;98m >=[0m [0;30;48;2;141;160;203m else[0m [0;30;48;2;231;138;195m:[0m [0;30;48;2;166;216;84m two[0m [0;30;48;2;255;217;47m tabs[0m [0;30;48;2;102;194;165m:"[0m [0;30;48;2;252;141;98m   [0m [0;30;48;2;141;160;203m "[0m [0;30;48;2;231;138;195m Three[0m [0;30;48;2;166;216;84m tabs[0m [0;30;48;2;255;217;47m:[0m [0;30;48;2;102;194;165m "[0m [0;30;48;2;252;141;98m      [0m [0;30;48;2;141;160;203m "
    [0m [0;30;48;2;231;138;195m12[0m [0;30;48;2;166;216;84m.[0m [0;30;48;2;255;217;47m0[0m [0;30;48;2;102;194;165m*[0m [0;30;48;2;252;141;98m50[0m [0;30;48;2;141;160;203m=[0m [0;30;48;2;231;138;195m600[0m [0;30;48;2;166;216;84m
    [0m 

### Optional Models to Explore

You can also explore the tokenization strategy of other models. The following is a suggested list. Make sure to consider the following features when you're doing your comparison:
- Vocabulary length
- Special tokens
- Tokenization of the tabs, special characters and special keywords

**gpt2**


```python
show_tokens(text, "gpt2")
```


    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]


    Vocab length: 50257
    [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mEnglish[0m [0;30;48;2;141;160;203m and[0m [0;30;48;2;231;138;195m CAP[0m [0;30;48;2;166;216;84mITAL[0m [0;30;48;2;255;217;47mIZ[0m [0;30;48;2;102;194;165mATION[0m [0;30;48;2;252;141;98m
    [0m [0;30;48;2;141;160;203m�[0m [0;30;48;2;231;138;195m�[0m [0;30;48;2;166;216;84m�[0m [0;30;48;2;255;217;47m �[0m [0;30;48;2;102;194;165m�[0m [0;30;48;2;252;141;98m�[0m [0;30;48;2;141;160;203m
    [0m [0;30;48;2;231;138;195mshow[0m [0;30;48;2;166;216;84m_[0m [0;30;48;2;255;217;47mt[0m [0;30;48;2;102;194;165mok[0m [0;30;48;2;252;141;98mens[0m [0;30;48;2;141;160;203m False[0m [0;30;48;2;231;138;195m None[0m [0;30;48;2;166;216;84m el[0m [0;30;48;2;255;217;47mif[0m [0;30;48;2;102;194;165m ==[0m [0;30;48;2;252;141;98m >=[0m [0;30;48;2;141;160;203m else[0m [0;30;48;2;231;138;195m:[0m [0;30;48;2;166;216;84m two[0m [0;30;48;2;255;217;47m tabs[0m [0;30;48;2;102;194;165m:"[0m [0;30;48;2;252;141;98m [0m [0;30;48;2;141;160;203m [0m [0;30;48;2;231;138;195m [0m [0;30;48;2;166;216;84m "[0m [0;30;48;2;255;217;47m Three[0m [0;30;48;2;102;194;165m tabs[0m [0;30;48;2;252;141;98m:[0m [0;30;48;2;141;160;203m "[0m [0;30;48;2;231;138;195m [0m [0;30;48;2;166;216;84m [0m [0;30;48;2;255;217;47m [0m [0;30;48;2;102;194;165m [0m [0;30;48;2;252;141;98m [0m [0;30;48;2;141;160;203m [0m [0;30;48;2;231;138;195m "[0m [0;30;48;2;166;216;84m
    [0m [0;30;48;2;255;217;47m12[0m [0;30;48;2;102;194;165m.[0m [0;30;48;2;252;141;98m0[0m [0;30;48;2;141;160;203m*[0m [0;30;48;2;231;138;195m50[0m [0;30;48;2;166;216;84m=[0m [0;30;48;2;255;217;47m600[0m [0;30;48;2;102;194;165m
    [0m 

**Flan-T5-small**


```python
show_tokens(text, "google/flan-t5-small")
```


    tokenizer_config.json:   0%|          | 0.00/2.54k [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.42M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/2.20k [00:00<?, ?B/s]


    Vocab length: 32100
    [0;30;48;2;102;194;165mEnglish[0m [0;30;48;2;252;141;98mand[0m [0;30;48;2;141;160;203mCA[0m [0;30;48;2;231;138;195mPI[0m [0;30;48;2;166;216;84mTAL[0m [0;30;48;2;255;217;47mIZ[0m [0;30;48;2;102;194;165mATION[0m [0;30;48;2;252;141;98m[0m [0;30;48;2;141;160;203m<unk>[0m [0;30;48;2;231;138;195m[0m [0;30;48;2;166;216;84m<unk>[0m [0;30;48;2;255;217;47mshow[0m [0;30;48;2;102;194;165m_[0m [0;30;48;2;252;141;98mto[0m [0;30;48;2;141;160;203mken[0m [0;30;48;2;231;138;195ms[0m [0;30;48;2;166;216;84mFal[0m [0;30;48;2;255;217;47ms[0m [0;30;48;2;102;194;165me[0m [0;30;48;2;252;141;98mNone[0m [0;30;48;2;141;160;203m[0m [0;30;48;2;231;138;195me[0m [0;30;48;2;166;216;84ml[0m [0;30;48;2;255;217;47mif[0m [0;30;48;2;102;194;165m=[0m [0;30;48;2;252;141;98m=[0m [0;30;48;2;141;160;203m>[0m [0;30;48;2;231;138;195m=[0m [0;30;48;2;166;216;84melse[0m [0;30;48;2;255;217;47m:[0m [0;30;48;2;102;194;165mtwo[0m [0;30;48;2;252;141;98mtab[0m [0;30;48;2;141;160;203ms[0m [0;30;48;2;231;138;195m:[0m [0;30;48;2;166;216;84m"[0m [0;30;48;2;255;217;47m"[0m [0;30;48;2;102;194;165mThree[0m [0;30;48;2;252;141;98mtab[0m [0;30;48;2;141;160;203ms[0m [0;30;48;2;231;138;195m:[0m [0;30;48;2;166;216;84m"[0m [0;30;48;2;255;217;47m"[0m [0;30;48;2;102;194;165m12.[0m [0;30;48;2;252;141;98m0[0m [0;30;48;2;141;160;203m*[0m [0;30;48;2;231;138;195m50[0m [0;30;48;2;166;216;84m=[0m [0;30;48;2;255;217;47m600[0m [0;30;48;2;102;194;165m[0m [0;30;48;2;252;141;98m</s>[0m 

**Starcoder 2 - 15B**


```python
show_tokens(text, "bigcode/starcoder2-15b")
```


    tokenizer_config.json:   0%|          | 0.00/7.88k [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/777k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/442k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.06M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/958 [00:00<?, ?B/s]


    Vocab length: 49152
    [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mEnglish[0m [0;30;48;2;141;160;203m and[0m [0;30;48;2;231;138;195m CAPITAL[0m [0;30;48;2;166;216;84mIZATION[0m [0;30;48;2;255;217;47m
    [0m [0;30;48;2;102;194;165m�[0m [0;30;48;2;252;141;98m�[0m [0;30;48;2;141;160;203m�[0m [0;30;48;2;231;138;195m [0m [0;30;48;2;166;216;84m�[0m [0;30;48;2;255;217;47m�[0m [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mshow[0m [0;30;48;2;141;160;203m_[0m [0;30;48;2;231;138;195mtokens[0m [0;30;48;2;166;216;84m False[0m [0;30;48;2;255;217;47m None[0m [0;30;48;2;102;194;165m elif[0m [0;30;48;2;252;141;98m ==[0m [0;30;48;2;141;160;203m >=[0m [0;30;48;2;231;138;195m else[0m [0;30;48;2;166;216;84m:[0m [0;30;48;2;255;217;47m two[0m [0;30;48;2;102;194;165m tabs[0m [0;30;48;2;252;141;98m:"[0m [0;30;48;2;141;160;203m   [0m [0;30;48;2;231;138;195m "[0m [0;30;48;2;166;216;84m Three[0m [0;30;48;2;255;217;47m tabs[0m [0;30;48;2;102;194;165m:[0m [0;30;48;2;252;141;98m "[0m [0;30;48;2;141;160;203m      [0m [0;30;48;2;231;138;195m "[0m [0;30;48;2;166;216;84m
    [0m [0;30;48;2;255;217;47m1[0m [0;30;48;2;102;194;165m2[0m [0;30;48;2;252;141;98m.[0m [0;30;48;2;141;160;203m0[0m [0;30;48;2;231;138;195m*[0m [0;30;48;2;166;216;84m5[0m [0;30;48;2;255;217;47m0[0m [0;30;48;2;102;194;165m=[0m [0;30;48;2;252;141;98m6[0m [0;30;48;2;141;160;203m0[0m [0;30;48;2;231;138;195m0[0m [0;30;48;2;166;216;84m
    [0m 

**Phi-3**


```python
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")
```


    tokenizer_config.json:   0%|          | 0.00/3.44k [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.94M [00:00<?, ?B/s]



    added_tokens.json:   0%|          | 0.00/306 [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/599 [00:00<?, ?B/s]


    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


    Vocab length: 32011
    [0;30;48;2;102;194;165m[0m [0;30;48;2;252;141;98m
    [0m [0;30;48;2;141;160;203mEnglish[0m [0;30;48;2;231;138;195mand[0m [0;30;48;2;166;216;84mC[0m [0;30;48;2;255;217;47mAP[0m [0;30;48;2;102;194;165mIT[0m [0;30;48;2;252;141;98mAL[0m [0;30;48;2;141;160;203mIZ[0m [0;30;48;2;231;138;195mATION[0m [0;30;48;2;166;216;84m
    [0m [0;30;48;2;255;217;47m�[0m [0;30;48;2;102;194;165m�[0m [0;30;48;2;252;141;98m�[0m [0;30;48;2;141;160;203m�[0m [0;30;48;2;231;138;195m[0m [0;30;48;2;166;216;84m�[0m [0;30;48;2;255;217;47m�[0m [0;30;48;2;102;194;165m�[0m [0;30;48;2;252;141;98m
    [0m [0;30;48;2;141;160;203mshow[0m [0;30;48;2;231;138;195m_[0m [0;30;48;2;166;216;84mto[0m [0;30;48;2;255;217;47mkens[0m [0;30;48;2;102;194;165mFalse[0m [0;30;48;2;252;141;98mNone[0m [0;30;48;2;141;160;203melif[0m [0;30;48;2;231;138;195m==[0m [0;30;48;2;166;216;84m>=[0m [0;30;48;2;255;217;47melse[0m [0;30;48;2;102;194;165m:[0m [0;30;48;2;252;141;98mtwo[0m [0;30;48;2;141;160;203mtabs[0m [0;30;48;2;231;138;195m:"[0m [0;30;48;2;166;216;84m  [0m [0;30;48;2;255;217;47m"[0m [0;30;48;2;102;194;165mThree[0m [0;30;48;2;252;141;98mtabs[0m [0;30;48;2;141;160;203m:[0m [0;30;48;2;231;138;195m"[0m [0;30;48;2;166;216;84m     [0m [0;30;48;2;255;217;47m"[0m [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98m1[0m [0;30;48;2;141;160;203m2[0m [0;30;48;2;231;138;195m.[0m [0;30;48;2;166;216;84m0[0m [0;30;48;2;255;217;47m*[0m [0;30;48;2;102;194;165m5[0m [0;30;48;2;252;141;98m0[0m [0;30;48;2;141;160;203m=[0m [0;30;48;2;231;138;195m6[0m [0;30;48;2;166;216;84m0[0m [0;30;48;2;255;217;47m0[0m [0;30;48;2;102;194;165m
    [0m 

**Qwen2 - Vision-Language Model**


```python
show_tokens(text, "Qwen/Qwen2-VL-7B-Instruct")
```


    tokenizer_config.json:   0%|          | 0.00/4.19k [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/2.78M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/1.67M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/7.03M [00:00<?, ?B/s]


    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


    Vocab length: 151657
    [0;30;48;2;102;194;165m
    [0m [0;30;48;2;252;141;98mEnglish[0m [0;30;48;2;141;160;203m and[0m [0;30;48;2;231;138;195m CAPITAL[0m [0;30;48;2;166;216;84mIZATION[0m [0;30;48;2;255;217;47m
    [0m [0;30;48;2;102;194;165m🎵[0m [0;30;48;2;252;141;98m �[0m [0;30;48;2;141;160;203m�[0m [0;30;48;2;231;138;195m�[0m [0;30;48;2;166;216;84m
    [0m [0;30;48;2;255;217;47mshow[0m [0;30;48;2;102;194;165m_tokens[0m [0;30;48;2;252;141;98m False[0m [0;30;48;2;141;160;203m None[0m [0;30;48;2;231;138;195m elif[0m [0;30;48;2;166;216;84m ==[0m [0;30;48;2;255;217;47m >=[0m [0;30;48;2;102;194;165m else[0m [0;30;48;2;252;141;98m:[0m [0;30;48;2;141;160;203m two[0m [0;30;48;2;231;138;195m tabs[0m [0;30;48;2;166;216;84m:"[0m [0;30;48;2;255;217;47m   [0m [0;30;48;2;102;194;165m "[0m [0;30;48;2;252;141;98m Three[0m [0;30;48;2;141;160;203m tabs[0m [0;30;48;2;231;138;195m:[0m [0;30;48;2;166;216;84m "[0m [0;30;48;2;255;217;47m      [0m [0;30;48;2;102;194;165m "
    [0m [0;30;48;2;252;141;98m1[0m [0;30;48;2;141;160;203m2[0m [0;30;48;2;231;138;195m.[0m [0;30;48;2;166;216;84m0[0m [0;30;48;2;255;217;47m*[0m [0;30;48;2;102;194;165m5[0m [0;30;48;2;252;141;98m0[0m [0;30;48;2;141;160;203m=[0m [0;30;48;2;231;138;195m6[0m [0;30;48;2;166;216;84m0[0m [0;30;48;2;255;217;47m0[0m [0;30;48;2;102;194;165m
    [0m 

<p style="background-color:#f2f2ff; padding:15px; border-width:3px; border-color:#e2e2ff; border-style:solid; border-radius:6px"> ⬇
&nbsp; <b>Download Notebooks:</b> If you'd like to donwload the notebook: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
