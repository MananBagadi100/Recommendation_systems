import numpy as np
def get_collab_recommendation(product_id, cosine_item_similarity, user_item_matrix, top_r=5):
    ##for debugging
    print("🔥 product_id:", product_id)
    print("🔥 type:", type(product_id))
    print("🔥 Matrix column type:", type(user_item_matrix.columns[0]))
    print("🔥 Columns preview:", list(user_item_matrix.columns[:5]))
    print('columns dtype of user item matrix',user_item_matrix.columns.dtype)
    recommendations = {}
    
    
    if product_id not in user_item_matrix.columns:
        recommendations['1'] = f'Product {product_id} not found'
    else:
        product_index = user_item_matrix.columns.get_loc(product_id)
        #debugging
        print("🔥 product_index:", product_index)
        print("🔥 cosine row:", cosine_item_similarity[product_index][:5])

        product_similarity = cosine_item_similarity[product_index]

        print("product_index:", product_index) #for debugging

        product_similarity_sorted = np.argsort(product_similarity)[::-1]
        product_similarity_sorted = product_similarity_sorted[1:top_r+1]


        for i in product_similarity_sorted:
            product_id_similar = user_item_matrix.columns[i]
            similarity_score = product_similarity[i]
            recommendations[product_id_similar] = similarity_score

    print('THe recommendations are : ', recommendations)
    return recommendations
