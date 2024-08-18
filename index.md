
---
layout: default
---

# Hate Speech Detection Using AI: A Comprehensive Approach with BERT, LLMs, and Logistic Regression

## Introduction

In recent years, the proliferation of hate speech on social media platforms has become a significant concern. Detecting and mitigating such harmful content is crucial for maintaining a healthy online environment. With advancements in AI, particularly in Natural Language Processing (NLP), it's possible to automate the detection of hate speech with high accuracy. In this project, In this project, we explore multiple approaches, including traditional machine learning techniques like Bag of Words and TF-IDF with Logistic Regression and advanced models like BERT and GPT-3.5, to build a robust system for hate speech detection.

## Project Overview

### Objective

The primary goal of this project is to develop an AI-driven system capable of accurately detecting hate speech in text data. By experimenting with both traditional machine learning methods and state-of-the-art models, we aim to enhance detection accuracy and create a model that can be applied in real-world scenarios, where understanding the context and nuances of language is crucial.

### Dataset

For this project, we used the "Hate Speech and Offensive Language Dataset" available on [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset). This dataset contains **24,783 tweets labeled as "hate speech(represented as 0)," "offensive language(respresented as 1)," or "neither(represenred as 2)"**. 

### Methodology

#### 1. Explorartory Daat Analysis(EDA)
Before diving into model training, it was essential to understand the data. We conducted an extensive Exploratory Data Analysis (EDA) to examine the distribution of classes and the common words associated with each class. This step helped us gain insights into the characteristics of hate speech and offensive language.

We studied the class distribution for three different classes present in the dataset which are shown in the figure below.

![Image](https://github.com/women-in-ai-ireland/June-2024-Group-003/blob/WAI_blog/images/class_dis_1.png)

Additionally, the dataset exhibited extensive overlap in the words contained in three categories as shown beloww. **WARNIG: offensive language**

ADD WORD CLOUD

The existing dataset clearly showed bias towards the offensive/hate tweets and we decided to remedy that by augmenting with another dataset. In addition to that, we decided to build a binary classifier and merged the hate and offensive tweets into one class.

The second dataset used was titled "Twitter Tweets Sentiment Data" and is available on [Kaggle](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset). This dataset had 'positive', 'neutral' and 'neagtive' tweets. We only retained the 'positive' and 'neutral' tweets to mitigate the class imbalance in our original dataset as 'negative' label data can be construed as offensive. 


#### 2. Data Cleaning & Preprocessing

**Key Preprocessing Steps:**
*  **Text Cleaning:** Removed special characters, URLs, Twitter handles, and stop words to ensure the models focused on the core content. We also did a frequency analysis on the most common occuring words and removed words which would not aid the model in identifying hate speech(eg: 'got', 'day, 'just'). 
  
*  **Tokenization and Lemmatization:** Converted text into tokens and applied lemmatization to reduce words to their base forms, enhancing the model's ability to generalize.

ADD WORD CLOUD, FREQUENCY ANALYSIS PLOT AND MOST COMMON WORDS AFTER CLEANING

#### 3. Traditional Machine Learning Approach

**Logistic Regression with Bag of Words**

As a baseline, we implemented a traditional machine learning approach using Logistic Regression. This method is simple yet effective for binary classification tasks. After vectorizing the text data using techniques like TF-IDF and Bag of Words, we trained a Logistic Regression model to classify tweets.

The Logistic Regression model provided a solid foundation and baseline accuracy, helping us understand the complexity of the task and the need for more advanced models.

**Logistic Regression with TF-IDF**

#### 4. Advanced Model Deployment

**BERT Fine-Tuning**

We fine-tuned a pre-trained BERT model on our dataset. BERT's ability to understand the context within text made it an excellent choice for this task. After training, the BERT model provided predictions that were directly compared to the Logistic Regression outputs to assess the performance improvement.

#### 5. Model Evaluation

Each model was evaluated on a validation set, with performance metrics such as accuracy, precision, recall, and F1-score being computed. This step was crucial in understanding the effectiveness of our approaches and identifying areas for further improvement.

### Results and Discussion

The results from our experiments showed that while the Logistic Regression model provided a strong baseline, the BERT model significantly improved classification accuracy due to its deep understanding of context. GPT-4, even without fine-tuning, offered valuable insights and competitive performance, demonstrating the power of large language models in NLP tasks.

### Challenges


### Conclusion


### Future Work

Looking ahead, we plan to explore a meta-model approach that combines the predictions from multiple models, including Logistic Regression, BERT, and GPT-4. This ensemble technique could potentially enhance the overall performance by leveraging the strengths of each model.

Additionally, applying this model to other datasets or even live social media data could provide more insights and opportunities for refinement. We also aim to explore other LLMs and advanced ensemble techniques to further improve the model's performance.

### Acknowledgements

Special thanks to our mentor, [Nabanita Roy](https://www.linkedin.com/in/nabanita-roy/), for guiding us through this project and to the Women in AI Ireland community for providing the resources and support necessary to bring this project to life.
