Lab Assignment - 1
Title: Data Scraping and Preprocessing using Python and Scrapy

Name: Keval Patodiya
ID: 202618056

## Project Overview

This project demonstrates a complete data analysis workflow, including **web scraping**, **data cleaning**, **preprocessing**, **exploratory data analysis (EDA)**, and **visualization**. The data was collected from the **Books to Scrape** website using the Scrapy framework and analyzed using Python.

The objective of the project is to extract book information from multiple web pages, prepare the dataset for analysis, and generate meaningful insights through statistical analysis and visualizations.

---

## Objectives

* Scrape book information from the *Books to Scrape* website.
* Store the extracted data in CSV format.
* Clean and preprocess the collected dataset.
* Perform exploratory data analysis (EDA).
* Visualize trends and patterns in the data.
* Draw insights from the scraped dataset.

---

## Technologies Used

* Python 3
* Scrapy
* Pandas
* NumPy
* Matplotlib
* WordCloud
* Jupyter Notebook

---

## Dataset

The dataset was scraped from:

**https://books.toscrape.com/**

The spider crawls the first **five catalogue pages** and visits each book's detail page to collect information.

### Attributes Collected

* Title
* Category
* Price
* Rating
* Availability
* Product Description
* UPC
* Number of Reviews
* Product URL

---

## Project Structure

```text
202618008_NandiniPipaliya_DS605/
│
├── books.csv
├── scrapy.cfg
├── README.md
├── Lab1_DataScraping_Preprocessing.ipynb
│
└── bookscraper/
    ├── items.py
    ├── pipelines.py
    ├── settings.py
    └── spiders/
        └── books.py
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the required packages:

```bash
pip install scrapy pandas matplotlib wordcloud numpy jupyter
```

---

## Running the Scraper

Generate the dataset by running:

```bash
scrapy crawl books -O books.csv
```

The command creates **books.csv** containing the scraped data.

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed duplicate records.
* Handled missing values.
* Replaced missing descriptions with **"Not Available"**.
* Converted prices into numeric values.
* Converted ratings into numerical format.
* Standardized text columns.
* Checked data types and cleaned inconsistencies.

---

## Exploratory Data Analysis

The notebook includes analyses such as:

* Distribution of book prices
* Rating distribution
* Category-wise book counts
* Average price by category
* Price vs. Rating relationship
* Availability analysis
* Word Cloud of book descriptions

---

## Key Insights

* Fiction and literature categories contain a large number of books.
* Book prices are spread across a moderate price range.
* Higher ratings do not necessarily correspond to higher prices.
* Some categories have noticeably higher average prices than others.
* Most books are available in stock.
* Product descriptions reveal common themes through word cloud visualization.

---

## Output

The project produces:

* Scraped dataset (**books.csv**)
* Cleaned dataset
* Exploratory analysis
* Statistical summaries
* Data visualizations
* Word Cloud
* Insights and conclusions

---

## Learning Outcomes

Through this project, the following skills were demonstrated:

* Web scraping using Scrapy
* Data collection from websites
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization using Matplotlib
* Feature exploration and interpretation
* Python programming for data analysis

---
