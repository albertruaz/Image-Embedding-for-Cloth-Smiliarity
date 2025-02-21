import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_connector import DBConnector
from db.vector_db_connector import VectorDBConnector
from model.mediapipe_embedding_model import MediaPipeEmbeddingModel
import argparse

def main():
    # Input form: [(269337, 'https://~'), (269338, 'https://~'), ~]
    # example
    product_datas = [(269337, 'https://kuzyepey7184.edge.naverncp.com//product/unhashed/9b7f0d56-a60f-4cb1-996e-6846b680139b--1528276518')]
    
    # Embedding
    model = MediaPipeEmbeddingModel(model_name="embedder.tflite")
    embedding = model.embed_batch(product_datas, (224, 224))
    
    # Output form: [{'product_id': 269337, 'image_vector': [백터]},{'product_id': ~, 'image_vector': [~]}]
    print(embedding)

if __name__ == "__main__":
    main()
