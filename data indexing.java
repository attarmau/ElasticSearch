from Elasticsearch import Elasticsearch

# create an Elasticsearch client object
es = Elasticsearch()

# define the index and document type
index_name = "my_index"
doc_type = "my_doc_type"

# define the document to be indexed
doc = {
    "title": "Example Document",
    "content": "This is an example document to be indexed in Elasticsearch."
}

# index the document
res = es.index(index=index_name, doc_type=doc_type, body=doc)

# print the response
print(res)
