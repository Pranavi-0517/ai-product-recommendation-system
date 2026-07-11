import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import json
from collections import defaultdict

class ProductRecommender:
    def __init__(self):
        self.products_df = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.user_item_matrix = None
        self.product_similarity = None
        
    def load_products(self, products_data):
        """Load products from JSON data"""
        self.products_df = pd.DataFrame(products_data)
        # Create feature text for content-based filtering
        self.products_df['features'] = self.products_df.apply(
            lambda x: f"{x['name']} {x['category']} {' '.join(x['tags'])} {x['description']}", 
            axis=1
        )
        self._build_content_based_model()
        return self.products_df
    
    def _build_content_based_model(self):
        """Build content-based recommendation model"""
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = tfidf.fit_transform(self.products_df['features'])
        # Calculate similarity matrix
        self.product_similarity = cosine_similarity(self.tfidf_matrix)
        
    def get_content_based_recommendations(self, product_id, top_n=5):
        """Get recommendations based on product similarity"""
        try:
            idx = self.products_df[self.products_df['id'] == product_id].index[0]
            similarity_scores = list(enumerate(self.product_similarity[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            # Get top N similar products (excluding itself)
            similarity_scores = similarity_scores[1:top_n+1]
            product_indices = [i[0] for i in similarity_scores]
            return self.products_df.iloc[product_indices].to_dict('records')
        except:
            return []
    
    def train_collaborative_model(self, interactions_data):
        """Train collaborative filtering model"""
        if not interactions_data:
            return
        
        # Create user-item matrix
        df = pd.DataFrame(interactions_data)
        self.user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating',
            fill_value=0
        )
        
        # Apply SVD for dimensionality reduction
        if len(self.user_item_matrix) > 1 and len(self.user_item_matrix.columns) > 1:
            self.svd_model = TruncatedSVD(n_components=min(10, len(self.user_item_matrix)-1, len(self.user_item_matrix.columns)-1))
            self.svd_model.fit(self.user_item_matrix)
    
    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Get recommendations based on user similarity"""
        if self.svd_model is None or user_id not in self.user_item_matrix.index:
            # Fallback to popular items
            return self.get_popular_products(top_n)
        
        # Get user's latent factors
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        user_latent = self.svd_model.transform(user_vector)
        
        # Predict ratings for all products
        product_latent = self.svd_model.components_.T
        predictions = np.dot(user_latent, product_latent.T).flatten()
        
        # Get top N recommendations
        product_ids = self.user_item_matrix.columns
        pred_df = pd.DataFrame({
            'product_id': product_ids,
            'pred_rating': predictions
        })
        
        # Filter out products the user has already interacted with
        user_interactions = self.user_item_matrix.iloc[user_idx]
        interacted_products = user_interactions[user_interactions > 0].index.tolist()
        pred_df = pred_df[~pred_df['product_id'].isin(interacted_products)]
        
        # Get top N
        top_products = pred_df.sort_values('pred_rating', ascending=False).head(top_n)
        return self.products_df[self.products_df['id'].isin(top_products['product_id'])].to_dict('records')
    
    def get_hybrid_recommendations(self, user_id=None, product_id=None, top_n=5):
        """Combine content-based and collaborative filtering"""
        if user_id and product_id:
            # Hybrid for specific user viewing a product
            cb_recs = self.get_content_based_recommendations(product_id, top_n=top_n*2)
            if self.svd_model is not None:
                cf_recs = self.get_collaborative_recommendations(user_id, top_n=top_n*2)
                # Combine and deduplicate
                all_recs = cb_recs + cf_recs
                seen_ids = set()
                unique_recs = []
                for rec in all_recs:
                    if rec['id'] not in seen_ids and rec['id'] != product_id:
                        seen_ids.add(rec['id'])
                        unique_recs.append(rec)
                return unique_recs[:top_n]
            return cb_recs[:top_n]
        elif user_id:
            # User-based recommendations
            return self.get_collaborative_recommendations(user_id, top_n)
        elif product_id:
            # Product-based recommendations
            return self.get_content_based_recommendations(product_id, top_n)
        else:
            # Default: popular products
            return self.get_popular_products(top_n)
    
    def get_popular_products(self, top_n=5):
        """Get most popular products"""
        popular = self.products_df.sort_values('rating', ascending=False).head(top_n)
        return popular.to_dict('records')
    
    def get_recommendations_for_user_history(self, user_history, top_n=5):
        """Get recommendations based on user's browsing/purchase history"""
        if not user_history:
            return self.get_popular_products(top_n)
        
        # Get product IDs from history
        product_ids = [item['product_id'] for item in user_history]
        # Get recommendations for each viewed product
        all_recs = []
        for pid in product_ids:
            recs = self.get_content_based_recommendations(pid, top_n=3)
            all_recs.extend(recs)
        
        # Deduplicate and rank
        seen_ids = set()
        unique_recs = []
        for rec in all_recs:
            if rec['id'] not in seen_ids and rec['id'] not in product_ids:
                seen_ids.add(rec['id'])
                unique_recs.append(rec)
        
        return unique_recs[:top_n]