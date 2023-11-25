# Cross-lingual Books Alignment with Transformers and OpenVINO™

Cross-lingual text alignment is the task of matching sentences in a pair of texts that are translations of each other. In this notebook, you'll learn how to use a deep learning model to create a parallel book in English and German.

This method not only helps you learn languages but also provides parallel texts that can be used to train machine translation models. This is particularly useful if one of the languages is low-resource or you don't have enough data to train a full-fledged translation model.

The notebook shows how to accelerate the most computationally expensive part of the pipeline - getting vectors from sentences - using the OpenVINO™ framework.

## Pipeline

The notebook guides you through the entire process of creating a parallel book: from obtaining raw texts to building a visualization of aligned sentences. Here is the pipeline diagram:

![](https://user-images.githubusercontent.com/51917466/254582697-18f3ab38-e264-4b2c-a088-8e54b855c1b2.png)

Visualizing the result allows you to identify areas for improvement in the pipeline steps, as indicated in the diagram.

## Prerequisites

- `requests` - for getting books
- `pysbd` - for splitting sentences
- `transformers[torch]` and `openvino_dev` - for getting sentence embeddings
- `seaborn` - for alignment matrix visualization
- `ipywidgets` - for displaying HTML and JS output in the notebook

```bash
!pip install -q requests pysbd transformers[torch] "openvino_dev>=2023.0" seaborn ipywidgets
```

## Table of Contents
- [Get Books](#Get-Books-Uparrow)
- [Clean Text](#Clean-Text-Uparrow)
- [Split Text](#Split-Text-Uparrow)
- [Get Sentence Embeddings](#Get-Sentence-Embeddings-Uparrow)
    - [Optimize the Model with OpenVINO](#Optimize-the-Model-with-OpenVINO-Uparrow)
- [Calculate Sentence Alignment](#Calculate-Sentence-Alignment-Uparrow)
- [Postprocess Sentence Alignment](#Postprocess-Sentence-Alignment-Uparrow)
- [Visualize Sentence Alignment](#Visualize-Sentence-Alignment-Uparrow)
- [Speed up Embeddings Computation](#Speed-up-Embeddings-Computation-Uparrow)

## Get Books [$\Uparrow$](#Table-of-content:)

The first step is to get the books that we will be working with. For this notebook, we will use English and German versions of Anna Karenina by Leo Tolstoy. The texts can be obtained from the [Project Gutenberg site](https://www.gutenberg.org/). Since copyright laws are complex and differ from country to country, check the book's legal availability in your country. Refer to the Project Gutenberg Permissions, Licensing and other Common Requests [page](https://www.gutenberg.org/policy/permission.html) for more information.

Find the books on Project Gutenberg [search page](https://www.gutenberg.org/ebooks/) and get the ID of each book. To get the texts, we will pass the IDs to the [Gutendex](http://gutendex.com/) API.

```python
import requests

def get_book_by_id(book_id: int, gutendex_url: str = "https://gutendex.com/") -> str:
    book_metadata_url = gutendex_url + "/books/" + str(book_id)
    request = requests.get(book_metadata_url, timeout=30)
    request.raise_for_status()

    book_metadata = request.json()
    book_url = book_metadata["formats"]["text/plain"]
    return requests.get(book_url).text

en_book_id = 1399
de_book_id = 44956

anna_karenina_en = get_book_by_id(en_book_id)
anna_karenina_de = get_book_by_id(de_book_id)

# Let's check that we got the right books by showing a part of the texts:
print(anna_karenina_en[:1500])
```

## Clean Text [$\Uparrow$](#Table-of-content:)

The downloaded books may contain service information before and after the main text. The text might have different formatting styles and markup, for example, phrases from a different language enclosed in underscores for potential emphasis or italicization:

```python
import re
from contextlib import contextmanager
from tqdm.auto import tqdm

# ... (code omitted for brevity)

# Combine the cleaning functions into a single pipeline.
def clean_text(text: str) -> str:
    text_cleaning_pipeline = [
        remove_single_newline,
        unify_quotes,
        remove_markup,
    ]
    progress_bar = tqdm(text_cleaning_pipeline, disable=disable_tqdm)
    for clean_func in progress_bar:
        progress_bar.set_postfix_str(clean_func.__name__)
        text = clean_func(text)
    return text

chapter_1_en = clean_text(chapter_1_en)
chapter_1_de = clean_text(chapter_1_de)
```

## Split Text [$\Uparrow$](#Table-of-content:)

Dividing text into sentences is a challenging task in text processing. The problem is called [sentence boundary disambiguation](https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation), which can be solved using heuristics or machine learning models. This notebook uses a `Segmenter` from the `pysbd` library, which is initialized with an [ISO language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), as the rules for splitting text into sentences may vary for different languages:

```python
import pysbd

splitter_en = pysbd.Segmenter(language="en", clean=True)
splitter_de = pysbd.Segmenter(language="de", clean=True)

sentences_en = splitter_en.segment(chapter_1_en)
sentences_de = splitter_de.segment(chapter_1_de)

len(sentences_en), len(sentences_de)
```

## Get Sentence Embeddings [$\Uparrow$](#Table-of-content:)

The next step is to transform sentences into vector representations. Transformer encoder models, like BERT, provide high-quality embeddings but can be slow. Additionally, the model should support both chosen languages. Training separate models for each language pair can be expensive, so there are many models pre-trained on multiple languages simultaneously:

- [multilingual-MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [distiluse-base-multilingual-cased](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
- [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)
- [LaBSE](https://huggingface.co/rasa/LaBSE)

LaBSE stands for [Language-agnostic BERT Sentence Embedding](https://arxiv.org/pdf/2007.01852.pdf) and supports 109+ languages. It has the same architecture as the BERT model but has been trained on a different task: to produce identical embeddings for translation pairs.

![](https://user-images.githubusercontent.com/51917466

/12726179-a2b4b5ba-d33b-4ea3-afcc-61f88da7c1be.png)

You can use any of these models, but for this notebook, LaBSE will be used:

```python
from transformers import SentenceTransformer

model = SentenceTransformer("LaBSE")

# Calculate embeddings for English and German sentences
embeddings_en = model.encode(sentences_en, convert_to_tensor=True)
embeddings_de = model.encode(sentences_de, convert_to_tensor=True)

# Display the shape of the obtained embeddings
embeddings_en.shape, embeddings_de.shape
```

### Optimize the Model with OpenVINO [$\Uparrow$](#Table-of-content:)

Deep learning models are resource-intensive and can benefit from optimization to run efficiently on different hardware. The OpenVINO toolkit is a solution for optimizing and deploying models across various architectures.

Below is an example of how to use OpenVINO to optimize the LaBSE model:

```python
from openvino.inference_engine import IECore

# Load the LaBSE model
labse_model = SentenceTransformer("LaBSE", device="cpu")

# Convert the model to OpenVINO format
openvino_model_path = "labse_openvino"
labse_model.save(openvino_model_path, openvino=True)

# Create an Inference Engine core
ie = IECore()

# Read the optimized OpenVINO model
labse_openvino = ie.read_network(model=openvino_model_path + "/sentence_transformer.onnx")

# Load the network to the device
exec_net = ie.load_network(network=labse_openvino, device_name="CPU", num_requests=1)

# Calculate embeddings using OpenVINO
embeddings_en_openvino = calculate_embeddings_openvino(sentences_en, exec_net)
embeddings_de_openvino = calculate_embeddings_openvino(sentences_de, exec_net)

# Display the shape of the obtained embeddings
embeddings_en_openvino.shape, embeddings_de_openvino.shape
```

## Calculate Sentence Alignment [$\Uparrow$](#Table-of-content:)

Now that we have sentence embeddings for both English and German sentences, we can calculate the alignment score between each pair of sentences. The alignment score indicates how similar the sentences are to each other.

There are different methods to calculate alignment scores. One common method is to use the cosine similarity between sentence embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix between English and German embeddings
cosine_sim_matrix = cosine_similarity(embeddings_en, embeddings_de)

# Display the shape of the similarity matrix
cosine_sim_matrix.shape
```

## Postprocess Sentence Alignment [$\Uparrow$](#Table-of-content:)

The alignment matrix obtained from the cosine similarity scores can be postprocessed to improve the results. For example, you can use a threshold to filter out low similarity scores and adjust the threshold based on the distribution of scores:

```python
import numpy as np

# Set a threshold to filter out low similarity scores
threshold = 0.6

# Apply the threshold to obtain a binary alignment matrix
alignment_matrix = (cosine_sim_matrix > threshold).astype(np.int32)

# Display the shape of the alignment matrix
alignment_matrix.shape
```

## Visualize Sentence Alignment [$\Uparrow$](#Table-of-content:)

The final step is to visualize the sentence alignment. A heatmap is a common way to represent the alignment matrix:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the alignment matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(alignment_matrix, cmap="coolwarm", cbar_kws={'label': 'Cosine Similarity'})
plt.title('Sentence Alignment Matrix')
plt.xlabel('German Sentences')
plt.ylabel('English Sentences')
plt.show()
```

## Speed up Embeddings Computation [$\Uparrow$](#Table-of-content:)

Calculating sentence embeddings can be computationally expensive, especially when dealing with large texts. To speed up the process, you can use OpenVINO to optimize the model and accelerate inference. The following functions demonstrate how to calculate sentence embeddings using OpenVINO:

```python
# Define the function to calculate embeddings using OpenVINO
def calculate_embeddings_openvino(sentences, exec_net):
    # Convert sentences to embeddings using OpenVINO
    input_blob = next(iter(exec_net.inputs))
    output_blob = next(iter(exec_net.outputs))
    embeddings = []
    for sentence in tqdm(sentences, disable=disable_tqdm):
        input_data = {input_blob: sentence.unsqueeze(0).detach().cpu().numpy()}
        result = exec_net.infer(inputs=input_data)
        embeddings.append(torch.tensor(result[output_blob][0]))
    return torch.stack(embeddings)

# Calculate embeddings using OpenVINO for English and German sentences
embeddings_en_openvino = calculate_embeddings_openvino(sentences_en, exec_net)
embeddings_de_openvino = calculate_embeddings_openvino(sentences_de, exec_net)

# Display the shape of the obtained embeddings
embeddings_en_openvino.shape, embeddings_de_openvino.shape
```

This notebook provides a comprehensive guide to creating a parallel book in English and German using cross-lingual text alignment. Additionally, it demonstrates how to optimize the computation of sentence embeddings using the OpenVINO toolkit, resulting in faster inference times.
