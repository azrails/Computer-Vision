# Computer-Vision
<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#content">Content</a>
</p>

<div align="center">
  <a href="/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version">
</div>

## About

This repository contains Python notebooks and mini-projects on Computer Vision (CV).

Most content consists of:
- Homework assignments from HSE University courses
- Yandex training sessions
- Personal research projects exploring new CV approaches

## Installation

1. Install [Poetry](https://python-poetry.org/) (dependency management tool):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
2. Install dependencies
```bash
poetry install
```

## Content
<!-- (TEMPLATE)
### 📁 [Project Name](/path/to/directory)
- **Description**: Brief project summary (1-2 sentences)
- **Technologies**: Main technologies/libraries used
- **Key Features**: Core functionality or implemented methods -->

### 📁 [HW1 Filters-Detectors](./FiltersDetectors/FiltersDetectors.ipynb)
- **Description**: Study of the main tasks and methods of computer vision
- **Technologies**: Numpy, OpenCV
- **Key Features**: Corner / Edge detectors, Morphological transformations, Image preprocessing

### 📁 [HW3 Transfer-learning](./TransferLearning/homework3.ipynb)
- **Description**: The basic task is to try out different ways of training models (Fine-tune, Transfer-Learning).
- **Technologies**: PyTorch, torchvision
- **Key Features**: ResNet-50 / YOLO fine-tune

<!-- ### 📁 [GPT like transformer with Mixture of Experts](./gpt-moe/gpt_moe.ipynb)
- **Description**: Implementation GPT-like model with Mixture-of-Experts and Grouped Query Attention.
- **Technologies**: PyTorch
- **Key Features**: GPT transformer, Mixture of Experts (simple variant), Grouped Query Attention -->
