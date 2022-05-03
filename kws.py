class KeywordSpotting:
	'''
		Computes the affinity kernel propagation between two feature vectors
	'''
	class Affinity:
		def kernal_tracer(self, cost_mat, match_score): 
			r, c = cost_mat.shape
			match_coll=[]
			for rid in range(r-1, 0, -1): 
				for cid in range(c-1, 0, -1): 
					if(cost_mat[rid, cid]>=match_score):
						match = [] 
						match.append((rid,cid))
						cost_mat[rid,cid] =-1
						x = rid-1
						y = cid-1
						while(cost_mat[x,y]>=match_score): 
							match.append((x,y))
							cost_mat[x,y] =-1
							x -=1
							y -=1
						if(len(match)>2):
							match_coll.append(match)
			#print(match_coll)
			return match_coll
					
		 
		def affine_cost_matrix(self, query, target, match_score=1, gap_cost=0, threshold=0.5):
			'''
				Computes the affinity cost matrix for similarity computation
			'''
			tar_frames=[i for i in range(len(target))]
			H = np.zeros((len(query)+1, len(target)+1))
			for q in range(1,H.shape[0]): 
				for t in range(1,H.shape[1]): 
					q_q = query[q-1,:]
					t_t = target[t-1,:]
					cos_sim = 1-cosine(q_q, t_t) # compute the cosine similarity 
					#print(cos_sim)
					H[q,t] = cos_sim
			
			C = np.zeros((len(query) + 1, len(target) + 1), np.int)
			for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
				if(H[i,j]>=threshold):
					match =  match_score+C[i - 1, j - 1]
					C[i, j] = match
			
			cost_trace = np.copy(C)
			matches = self.kernal_tracer(cost_trace, match_score)
			return H, C, matches
		
		def get_sim_mat(self, query_feats, target_feats):
			sim_mat, cost_mat, matches_coord = self.affine_cost_matrix(query_feats,target_feats)
			return sim_mat, cost_mat, matches_coord
