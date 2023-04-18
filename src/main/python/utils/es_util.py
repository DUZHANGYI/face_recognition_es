import traceback

from elasticsearch import Elasticsearch, exceptions

from config import es_config

es = Elasticsearch(es_config.elasticsearch_url)


def create_index():
    mappings = {
        "properties": {
            "face_name": {
                "type": "keyword"
            },
            "face_encoding": {
                "type": "dense_vector",
                "dims": 128
            }
        }
    }
    # 创建索引
    response = es.indices.create(index=es_config.es_index, mappings=mappings)
    return response


def put_face(face_name, face_encoding):
    doc = {
        es_config.face_name_filed: face_name,
        es_config.face_encoding_filed: face_encoding
    }
    res = es.index(index=es_config.es_index, document=doc)
    return res


def search(face_encoding_list, min_score=0.92):
    script_query = _get_script_query(face_encoding_list, min_score)
    response = es.search(index=es_config.es_index, query=script_query, _source=es_config.face_name_filed, size=1)
    res = response['hits']['hits']
    if len(res) <= 0:
        return None
    hit = res[0]
    return hit['_source'][es_config.face_name_filed], hit['_score'] * 100


def batch_search(face_encoding_list, min_score=0.92, size=10):
    script_query = _get_script_query(face_encoding_list, min_score)
    names = []
    encodings = []
    scores = []
    try:
        response = es.search(index=es_config.es_index, query=script_query,
                             _source=[es_config.face_name_filed, es_config.face_encoding_filed],
                             size=size)
    except exceptions.RequestError:
        traceback.print_exc()
        return names, encodings, scores
    res = response['hits']['hits']
    if len(res) <= 0:
        return names, encodings, scores
    for hit in res:
        names.append(hit['_source'][es_config.face_name_filed])
        encodings.append(hit['_source'][es_config.face_encoding_filed])
        scores.append(hit['_score'] * 100)
    return names, encodings, scores


def _get_script_query(face_encoding_list, min_score=0.92):
    assert 0 <= min_score <= 1, "参数 min_score 必须是介于0和1之间的小数"
    # 使用 Script Score Query 搜索文档
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'face_encoding')",
                "params": {"query_vector": face_encoding_list}
            },
            "min_score": min_score
        }
    }
    return script_query
