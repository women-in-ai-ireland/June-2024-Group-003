
---
layout: default
---

# Hate Speech Detection Using AI: A Comprehensive Approach with BERT and Logistic Regression

## Introduction

In recent years, the proliferation of hate speech on social media platforms has become a significant concern. Detecting and mitigating such harmful content is crucial for maintaining a healthy online environment. With advancements in AI, particularly in Natural Language Processing (NLP), it's possible to automate the detection of hate speech with high accuracy. In this project, we explore multiple approaches, including traditional machine learning techniques like Bag of Words and TF-IDF with Logistic Regression, as well as advanced models like BERT, to build a robust system for hate speech detection.

## Project Overview

### Objective

The primary goal of this project is to develop an AI-driven system capable of accurately detecting hate speech in text data. By experimenting with both traditional machine learning methods and state-of-the-art models, we aim to enhance detection accuracy and create a model that can be applied in real-world scenarios, where understanding the context and nuances of language is crucial.

### Dataset

For this project, we used the "Hate Speech and Offensive Language Dataset" available on [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset). This dataset contains **24,783 tweets labeled as "hate speech(represented as 0)," "offensive language(respresented as 1)," or "neither(represenred as 2)"**. 

### Methodology

#### 1. Explorartory Daat Analysis(EDA)
Before diving into model training, it was essential to understand the data. We conducted an extensive Exploratory Data Analysis (EDA) to examine the distribution of classes and the common words associated with each class. This step helped us gain insights into the characteristics of hate speech and offensive language.

**Class Distribution**
We studied the class distribution for three different classes present in the dataset which are shown in the figure below.

![Image](https://github.com/women-in-ai-ireland/June-2024-Group-003/blob/WAI_blog/images/class_dis_1.png)

**Common Words Analysis**
Additionally, the dataset exhibited extensive overlap in the words contained in three categories as shown below. **WARNING: offensive language**

ADD WORD CLOUD

**Additional Data Augmentation**
The existing dataset clearly showed bias towards the offensive/hate tweets and we decided to remedy that by augmenting with another dataset. In addition to that, we decided to build a binary classifier and merged the hate and offensive tweets into one class.

The second dataset used was titled "Twitter Tweets Sentiment Data" and is available on [Kaggle](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset). This dataset had 'positive', 'neutral' and 'negative' tweets. We only retained the 'positive' and 'neutral' tweets to mitigate the class imbalance in our original dataset as 'negative' label data can be construed as offensive. 


#### 2. Data Cleaning & Preprocessing

To ensure our models could focus on the relevant content, we implemented several key preprocessing steps:
*  **Text Cleaning:** Removed special characters, URLs, Twitter handles, and stop words to ensure the models focused on the core content. We also did a frequency analysis on the most common occuring words and removed words which would not aid the model in identifying hate speech(eg: 'got', 'day, 'just'). 
  
*  **Tokenization and Lemmatization:** Converted text into tokens and applied lemmatization to reduce words to their base forms, enhancing the model's ability to generalize.

ADD WORD CLOUD, FREQUENCY ANALYSIS PLOT AND MOST COMMON WORDS AFTER CLEANING

#### 3. Traditional Machine Learning Approach

**Bag of Words**
Bag of Words (BoW) is one of the simplest methods of text representation in NLP. It converts text into a fixed-length vector of word occurrences, ignoring grammar and word order but capturing the frequency of individual words. Each word in a document is represented by its frequency or presence/absence.

**Term Frequency-Inverse Document Frequency**
TF-IDF is an extension of BoW that not only accounts for word frequency (Term Frequency, TF) but also considers how common or rare a word is across the entire dataset (Inverse Document Frequency, IDF). Words that are common across many documents (e.g., "the", "and") are down-weighted, while rare but important words get higher weights.

TF-IDF values adjust the word frequencies so that commonly used words in all documents are less impactful. This method helps in reducing the weight of commonly occurring words, giving more importance to rare but significant terms.

**Logistic Regression with BoW and TF-IDF**
As a baseline, we implemented Logistic Regression, a traditional machine learning approach, using both Bag of Words and TF-IDF. These methods are simple yet effective for binary classification tasks. After vectorizing the text data, we trained a Logistic Regression model to classify tweets.

This approach provided a solid foundation and baseline accuracy, helping us understand the complexity of the task and the need for more advanced models.

#### 4. Advanced Model Deployment

**BERT Fine-Tuning**
**BERT (Bidirectional Encoder Representations from Transformers)** is a revolutionary model in NLP, developed by Google. Unlike traditional models that process text sequentially, BERT reads text bidirectionally. This means it considers the context of a word based on the words that come before and after it, allowing BERT to understand the nuances and context of language far better than previous models.

**How BERT Works:**

**Pre-training:** BERT is pre-trained on a large corpus of text, learning to predict missing words in a sentence (Masked Language Model) and to predict if two sentences are sequential (Next Sentence Prediction). This pre-training gives BERT a deep understanding of language structure and context.

**Fine-tuning:** For specific tasks like hate speech detection, BERT can be fine-tuned on a smaller, task-specific dataset. During fine-tuning, BERT adjusts its parameters to better understand the nuances of the task at hand, such as identifying hate speech based on its training.

INSERT BERT ARCHITECTURE ILLUSTRATION

We fine-tuned a pre-trained BERT model on our dataset. BERT's ability to understand the context within text made it an excellent choice for this task. After training, the BERT model provided predictions that were directly compared to the Logistic Regression outputs to assess the performance improvement.

#### 5. Model Evaluation

We evaluated each model on a validation set drawn from our dataset. The Logistic Regression and BERT models both achieved an impressive performance, with around 98% accuracy on this internal validation set. However, our results showed that BERT's performance was similar to that of the Logistic Regression model. This outcome suggests that the complexities of our dataset, or perhaps its size and balance, limited the potential gains from using such an advanced model.

We also assessed the generalizability of our models by testing theme on an external dataset, the [Ethos Binary Dataset](https://huggingface.co/datasets/iamollas/ethos) from Hugging Face, which contains text labeled as either hate speech or not.

When tested on the Ethos dataset, the performance of both models dropped significantly, with accuracy decreasing to around 65%. This substantial drop highlights the challenge of deploying models trained on one dataset to a different one, especially in the nuanced task of hate speech detection where context and phrasing can vary widely.

This step was crucial in understanding the effectiveness of our approaches and identifying areas for further improvement.

### Results and Discussion

Our experiments demonstrated that while advanced models like BERT hold promise, traditional methods like Logistic Regression can perform equally well in certain scenarios. This finding suggests that for tasks with limited or imbalanced data, simpler models may be just as effective, if not more practical, due to their lower computational costs and ease of implementation. The significant drop in performance on the external Ethos dataset also underscores the importance of testing models on varied datasets to ensure their robustness and generalizability.

### Challenges
Throughout the project, we encountered challenges such as data imbalance, the difficulty of context understanding, and the high computational requirements of advanced models like BERT. Addressing these issues required careful data augmentation, model optimization, and leveraging external computational resources.

### Conclusion
This project demonstrated the effectiveness of combining traditional machine learning methods with advanced NLP models for hate speech detection. While the BERT model did not significantly outperform Logistic Regression, the insights gained from this project are invaluable. They highlight the importance of model selection based on the specific constraints and characteristics of the dataset.

The sharp contrast in performance between the internal validation set and the external Ethos dataset further emphasizes the need for diverse and representative training data in building robust models.

### Future Work

Looking ahead, we plan to explore **a meta-model approach that combines the predictions from multiple models**, including Logistic Regression and BERT. This ensemble technique could potentially enhance the overall performance by leveraging the strengths of each model.

Additionally, **applying this model to other datasets** or even live social media data could provide more insights and opportunities for refinement. We also aim to explore other large language models (LLMs) and advanced ensemble techniques to further improve the model's performance.

Another area of future work involves **integrating Generative AI (GenAI) to provide contextual explanations** for why a message was classified as offensive. Implementing such a system would involve training the GenAI model on labeled data where the rationale for classification is provided, enabling the model to learn patterns and generate meaningful explanations. This approach can also help in fine-tuning the model, making it more robust and aligned with human ethical standards.

As with any AI-driven system, it is crucial to **examine the ethical implications of our hate speech detection model**. One major concern is the potential for bias in the model's responses. Additionally, the implications of false negatives (i.e., instances where hate speech is not detected) are significant. If harmful content goes undetected, it can perpetuate toxicity and harm vulnerable individuals or groups.To address these issues, future work should focus on continuous monitoring and assessment of the model's outputs to detect and mitigate biases. This can involve re-training the model on more diverse datasets and incorporating fairness-aware algorithms to reduce the impact of any inherent biases.

### Acknowledgements

Special thanks to our mentor, [Nabanita Roy](https://www.linkedin.com/in/nabanita-roy/), for guiding us through this project and to the Women in AI Ireland community for providing the resources and support necessary to bring this project to life.
