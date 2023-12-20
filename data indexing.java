

public class ElasticsearchDocumentCount {
    public static void main(String[] args) {
        // Connect to Elasticsearch
        RestHighLevelClient client = new RestHighLevelClient(
                RestClient.builder(new HttpHost("localhost", 9200, "http")));

        try {
            // Prepare the search request
            SearchRequest searchRequest = new SearchRequest("products");
            SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
            searchSourceBuilder.query(QueryBuilders.matchAllQuery());
            searchSourceBuilder.size(0); // We only need the count, so set the size to 0
            searchRequest.source(searchSourceBuilder);

            // Execute the search request
            SearchResponse searchResponse = client.search(searchRequest);
        // Retrieve the count from the search response
            long totalDocuments = searchResponse.getHits().getTotalHits().value;
            System.out.println("Total documents in 'products' index: " + totalDocuments);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Close the Elasticsearch client connection
            try {
                client.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}

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
    "authors": ["put author name", "put another author name"],
    "keywords": ["deep learning", "image classification", "biomedical research"],
    "publication_date": "2022-01-01"
}

# index the document
res = es.index(index=index_name, doc_type=doc_type, body=doc)
# perform a search
query = {
    "query": {
        "match": {
            "keywords": ""
        }
    }
}

results = es.search(index=index_name, doc_type=doc_type, body=query)
for hit in results['hits']['hits']:
    print(hit['_source']['title'])
