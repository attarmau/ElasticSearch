from Elasticsearch import Elasticsearch

# create an Elasticsearch client object
es = Elasticsearch()

# define the index and document type
index_name = "my_index"
doc_type = "my_doc_type"
    
# define the mapping for the document type
mapping = {
    "properties": {
        "title": {
            "type": "text"
        },
        "abstract": {
            "type": "text"
        },
        "journal": {
            "type": "keyword"
        },
        "authors": {
            "type": "keyword"
        },
        "keywords": {
            "type": "keyword"
        },
        "publication_date": {
            "type": "date",
            "format": "yyyy-MM-dd"
        }
    }
}

# create the index with the mapping
es.indices.create(index=index_name, body={"mappings": {doc_type: mapping}})

# define a document to index
doc = {
    "title": "Deep Learning for Image Classification in Biomedical Research",
    "abstract": "This paper describes the use of deep learning techniques for image classification in biomedical research.",
    "journal": "Nature",
    "authors": ["John Smith", "Jane Doe"],
    "keywords": ["deep learning", "image classification", "biomedical research"],
    "publication_date": "2022-01-01"
}

# index the document
res = es.index(index=index_name, doc_type=doc_type, body=doc)

# print the response
print(res)
