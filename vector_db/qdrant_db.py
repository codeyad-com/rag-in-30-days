'''
Qdrant vector database module - qdrant-client-1.18.0
'''


import logging
from typing import List , Dict , Any , Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import  models

from settings import RAGConfig


logger = logging.getLogger(__name__)



class QdrantDB:

    def __init__(self, config: RAGConfig, collection_name:Optional[str] = None):
        self.config = config
        self.collection_name = collection_name or config.qdrant_collection

        self.client = QdrantClient(host = config.qdrant_host,
                                   port = config.qdrant_port)

        logger.info(f'Connected to Qdrant at {config.qdrant_host}:{config.qdrant_port}, collection: {self.collection_name}')
        logger.info(f'Using collection: {self.collection_name} ')

    def health_check(self) -> bool:

           try:
                self.client.get_collections()   
                return True
           except Exception as e:
                logger.error(f'Health check faild: {e}')      
                return False    

    def create_collection(self, force_recreate: bool = False) -> bool:
        collections = self.client.get_collections().collections

        if self.collection_name in collections:
              if force_recreate:
                   self.client.delete_collection(self.collection_name)
                   logger.info(f'Deleted exsisting collection: {self.collection_name}')
                   
                   return False

        self.client.create_collection(
             collection_name = self.collection_name,

             vectors_config = models.VectorParams(
                  size = self.config.qdrant_vector_size,
                  distance= models.Distance.COSINE)
        )

        logger.info(f'Created collection: {self.collection_name}')
        return True


    def collection_exists(self) -> bool:
         try:
              self.client.get_collection(self.collection_name)
              return True
         
         except Exception:
              return False


    def collection_info(self) -> Dict[str, Any]:
         try : 
              info = self.client.get_collection(self.collection_name) 
              return  {
                       'point_count': info.points_count,
                       'status':info.status

                  }
         except Exception as e:
              logger.error(f'Failed to get collection info: {e}')
              return {}


    def delete_all(self) -> bool:
         try:
              self.client.delete_collection(self.collection_name)
              logger.info(f'Deleted collection: {self.collection_name}')
              return True

         except Exception as e:
              logger.error(f'Failed to delete: {e}')
              return False




    def add_vectors(
              self,
              vectors:np.ndarray,
              payloads: List[Dict[str,Any]],
              batch_size: int =64
    )-> int:

        if not self.collection_exists():
             raise ValueError(f'collection {self.collection_name} does not exists.')
         

        if vectors.shape[0] != len(payloads):
            raise ValueError('mismatch between vectors and payloads count')


        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0 , 1 , norms)
        vectors = vectors/norms


        vectors_list = vectors.astype(np.float32).tolist()

        info = self.collection_info()
        start_id = info.get('points_count',0)
        ids = list(range(start_id, start_id + len(vectors_list)))

        # insert in batches
        total_inserted = 0
        for i in range(0,len(vectors_list), batch_size):
             self.client.upsert(

                  collection_name = self.collection_name,

                  points = models.Batch(
                       ids = ids[i: i + batch_size],
                       vectors= vectors_list[i: i + batch_size],
                       payloads = payloads[i: i+batch_size]
                  )
             )
             total_inserted+= len(vectors_list[i:i+batch_size])

        logger.info(f'inserted {total_inserted} vectors in to the {self.collection_name}')


    def search(
              self,
              query_vector: np.ndarray,
              top_k : int = 5,
              min_score: float = 0.0,

              )-> List[Dict[str, Any]]:

         if not self.collection_exists():
              return []

         query_vector = query_vector / np.linalg.norm(query_vector)

         results = self.client.query_points(
              collection_name  = self.collection_name,
              query = query_vector.tolislt(),
              limit= top_k,
              score_threshold = min_score,
              with_payload = True
         )

         formatted = []
         for result in results.points:
              formatted.append(
                   {
                        'id':result.id,
                        'score':result.score,
                        'text':result.payload.get('text',''),
                        'metadata':{
                             'doc_id':result.payload.get('doc_id',''),
                             'page_number':result.payload.get('page_number',0),
                             'source': result.payload.get('source','')
                        },
                   }
              )
         return formatted





        