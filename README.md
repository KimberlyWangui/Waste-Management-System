# EcoSort Intelligent Waste Management Assistant

An integrated AI system built for Metro City's waste management department that helps residents correctly sort waste through three complementary models: image-based classification, text-based classification, and retrieval-augmented recycling instruction generation.

## Overview

This project combines three machine learning components into a single assistant:

1. **CNN Image Classifier** — identifies waste material category from a photo
2. **Text Classifier** — identifies waste category from a written description
3. **RAG Instruction Generator** — retrieves relevant city policy documents and generates specific recycling instructions for the identified category

Either an image or a text description can be given as input, and the system returns a predicted waste category, a confidence score (image path only), generated recycling instructions, and the source policy documents that informed those instructions.

## Project Structure

```
waste-management-system/
├── data/
│   └── realwaste-main/
│       └── RealWaste/
│           ├── Cardboard/
│           ├── Food Organics/
│           ├── Glass/
│           ├── Metal/
│           ├── Miscellaneous Trash/
│           ├── Paper/
│           ├── Plastic/
│           ├── Textile Trash/
│           └── Vegetation/
├── waste_descriptions.csv
├── waste_policy_documents.json
└── waste_management_summative.ipynb
```

## Dataset

**RealWaste** — 4,752 images across 9 waste categories, sourced from the Whyte's Gully Waste and Resource Recovery facility in Wollongong, NSW, Australia. All images are a uniform 524×524 RGB, resized to 224×224 for model input. Category counts range from 318 (Textile Trash) to 921 (Plastic), so the dataset is moderately imbalanced.

> Licensed under CC BY-NC-SA 4.0. See original paper: [RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633).

**waste_descriptions.csv** — 5,000 generated short text descriptions (avg. 4.9 words) labeled by category, each with an associated disposal instruction and material composition note.

**waste_policy_documents.json** — 14 generated municipal policy documents covering acceptable/non-acceptable items, collection methods, and preparation instructions per category.

## Models

| Component | Architecture | Notes |
|---|---|---|
| Image classification | MobileNetV3Small (ImageNet-pretrained, transfer learning) | Frozen base, custom classification head, then fine-tuned on the top 15 layers |
| Text classification | DistilBERT (fully fine-tuned) | All layers trainable |
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2`, loaded via plain `transformers` (mean pooling + normalization) | Avoids a `sentence-transformers`/`aiohttp` dependency conflict encountered in this environment |
| Instruction generation | `google/flan-t5-small` | Beam search decoding, RAG-grounded on retrieved policy/description documents |

### Why MobileNetV3Small instead of Large, and CPU instead of GPU

Training was done entirely on **CPU**. The available hardware had Intel integrated graphics, which does not support the CUDA operations TensorFlow and PyTorch require, and the installed TensorFlow version (2.10) is also the last release with native Windows GPU support (later versions require WSL2 for GPU acceleration on Windows). Given this constraint, MobileNetV3**Small** was chosen over Large for faster training, the CNN base was kept largely frozen with a short, shallow fine-tune rather than deep end-to-end training, and `flan-t5-small` was used for generation rather than a larger variant. Each CNN training epoch took roughly 40–65 seconds on this setup.

## Results

**Image classification:** 78.54% test accuracy after fine-tuning (up from 76.46% with the frozen base). Strongest categories: Metal, Paper, Vegetation, Textile Trash (F1 0.79–0.85). Weakest: Miscellaneous Trash (recall 0.46, an inherently ambiguous catch-all category) and Glass (recall 0.63, likely confused with Metal/Plastic due to shared reflective/transparent visual properties).

**Text classification:** 100% test accuracy. This is very likely a reflection of the synthetic dataset's small vocabulary (305 unique words) and short, template-like descriptions rather than exceptional model generalization — there were no genuinely ambiguous descriptions in the test set to challenge the classifier.

**RAG instruction generation:** Sampling-based decoding produced fluent but circular, low-content output (e.g. *"Plastic recycling is a recyclable product that can be recycled"*). Switching to beam search with `min_new_tokens`, `no_repeat_ngram_size=3`, and a more directive extraction-style prompt produced meaningfully better, retrieval-grounded output across all 9 categories.

A genuine contradiction was found between the two source document types: `waste_descriptions.csv` states plastic bags can be returned to supermarket collection points, while `waste_policy_documents.json` explicitly lists plastic bags as non-acceptable for standard recycling bins. The RAG system faithfully reproduces this conflict rather than resolving it — a reminder that retrieval-augmented generation is only as reliable as its source documents.

## Integration Architecture

A single entry point, `waste_management_assistant(input_data, input_type)`, dispatches to either the image or text classifier based on `input_type`, then routes the predicted category through a shared retrieval-and-generation pipeline:

```
image ──► CNN ──┐
                 ├──► category ──► retrieve_policies() ──► generate_instructions() ──► response
text  ──► DistilBERT ──┘
```

The response dictionary includes `waste_category`, `confidence` (image path only — the text path returns `None` rather than fabricating a confidence value), `recycling_instructions`, and `relevant_policy_documents` (the retrieved source chunks, included for transparency).

## Known Limitations

- CNN accuracy is capped by CPU training constraints and would likely improve with deeper fine-tuning on a GPU.
- The text classifier's near-perfect accuracy reflects dataset simplicity, not a validated real-world capability; genuinely messy resident-submitted text would be a harder test.
- `flan-t5-small` requires careful decoding-parameter and prompt tuning to produce usable output, and its generations are shorter and less fluent than a larger model would produce.
- Source documents contain at least one unresolved factual contradiction (plastic bag disposal) that a production system would need to reconcile before deployment.
- The system assigns a single waste category per input; genuinely mixed-material items (e.g. a fabric item with a metal zipper) receive one label with no indication that multiple disposal streams may apply.

## Requirements

```bash
pip install tensorflow==2.10.0 torch transformers scikit-learn pandas numpy matplotlib seaborn pillow
```

## Usage

```python
result = waste_management_assistant("path/to/image.jpg", input_type="image")
result = waste_management_assistant("greasy cardboard pizza box", input_type="text")

print(result['waste_category'])
print(result['recycling_instructions'])
```
