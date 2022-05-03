from som import SOM
class SOMPost: # SOM with periodic boundary condition
	def predict_labels(self, feats_path, model_path, network_size): 
		r, c = network_size
		som = SOM(r, c)
		som.load(model_path)
		feats_arr = np.load(feats_path)
		win_labels = []
		for i in range(0,len(feats_arr)): 
			feat_vec = feats_arr[i]
			win_neuron = som.winner(feat_vec)
			win_labels.append(win_neuron)
		win_label_arr = np.asarray(win_labels)
		return win_label_arr
		
	def compute_posterior(self, kernal_mat, vector, network_size): 
		rows, clmns = network_size
		
		res_pos_vec = np.zeros((rows,clmns))
		for r in range(rows): 
			for c in range(clmns): 
				weight_r_c = kernal_mat[r,c,:]
				cos_sim = 1-cosine(vector, weight_r_c)
				res_pos_vec[r,c] = cos_sim
		
					
		#print(res_pos_vec)
		return res_pos_vec.flatten()
	
	def generate_posterior(self, feats_path, model_path, network_size): 
		r, c = network_size
		som = SOM(r, c)
		som.load(model_path)
		feats_arr = np.load(feats_path)
		som_weight = som.map
		print(som_weight.shape)
		feats_coll = []
		mapped_feats = []
		r, c = feats_arr.shape
		for i in range(0,r): 
			feats_coll.append(feats_arr[i])
		feats_coll_array = np.asarray(feats_coll)
		X_train = feats_coll_array
		
				
		for i in range(len(X_train)): 
			post_vector = self.compute_posterior(som_weight,X_train[i,:],network_size)
			norm_post = post_vector / np.linalg.norm(post_vector)
			mapped_feats.append(norm_post)
		return np.array(mapped_feats)
