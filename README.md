
# Embedding Model with Instructions

## Current Progress

- **Datasets**
    - Implemented
        - [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
            - 25k Train / 25k Test
            - labels (0 - neg; 1 - pos)
        - [yelp_review_full](https://huggingface.co/datasets/yelp_review_full)
            - 650k Train / 50k Test
            - labels (1 star ... 5 star)
        - [arXiv-Abstract-Label-20k](https://huggingface.co/datasets/Voice49/arXiv-Abstract-Label-20k)
            - 10k Train / 10k Test
            - labels (8 Primary Categories: Math, CS,...)
    - TODO: 
        - Challenging datasets
            - [DBLP](https://github.com/angelosalatino/dblp-parser)
            - [Amazon reviews](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
            - Explore Datasets from Kaggle
- **Embedding Models**
    - Implemented:
        - Bert
        - Bert-Large
        - Instructor 
        - T5
        - GPT2 (Medium)
    - TODO:
        - Implement larger models from [MTEB scoreboard](https://github.com/embeddings-benchmark/mteb)
- **Instructions**
    - [Experiment Log](https://docs.google.com/spreadsheets/d/1iBDq7C59G6olf_of_sTF5oCY3Itj6_kImzeUl3XMpd8/edit?usp=sharing)
    - TODO: 
        - Effects of instructions: study how sensitive models are to instructions.
- **Evaluation**
    - SVM (Linear)
    - MLP

## Research Questions:
- Does the inclusion of instructions improve the embeddings? 
- How can prompt engineering be used to further enhance embedding models?

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/ZikunFu/Embedding-Model-with-Instructions.git
cd Embedding-Model-with-Instructions
```

### For Conda

```bash
conda create --name embed --file environment.yml
conda activate embed
```

### For Pip

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository.
2. Create and activate the environment.
3. Run the Jupyter Notebook.


## Acknowledgments

- Hugging Face for providing the pre-trained models.
