# Information-Retrival-Wikipedia-Search-Engine

This project introduces a search engine that retrieves information from the entire Wikipedia corpus using BM25, tf-idf, weights and cosine similarity. 

Throughout the project we useds python common libraries such as pandas, numpy and many more...

## NOTE: 
This project was written completely in Python and implemented using  JupyterNoteBook and GoogleColab.

## Project files:

### inverted_index_gcp- 
used to create an inverted index object.

### search_fronted- 
Used to create the server-side using flask, receive queries from clients and provide answers.

### search_backend- 
Contains all implementations of helper functions for search_fronted. 

### create_inv_indx_gcp- 
Contains all creations of all inverted indexes. Each index is based on a different part of the document (title, body and anchor) using the spark library.  

We evaluated our results using map@40 and running times, 

##here is our main results:
![image](https://user-images.githubusercontent.com/107557426/212560902-57fed7c7-aab4-444c-9123-13c74204ff1f.png)


## diagram of the engine we created:

![image](https://user-images.githubusercontent.com/107557426/212560811-d9bbfe09-88bc-4b4b-bf21-ef87f369535f.png)
