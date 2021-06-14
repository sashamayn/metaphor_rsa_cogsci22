import numpy as np
from scipy.stats import norm
import csv, math
import sys
'''

'''

def priors_setup():
	#animals as keys
	priors_anim = {}
	#adjectives as keys
	priors_adj = {}
	animals = []
	adjectives = []

	with open('MeanSDsInclHuman.csv') as f:
		csv_reader = csv.reader(f, delimiter=',')
		line_count = 0
		for row in csv_reader:
			# print(row)
			if line_count == 0:
				line_count+=1
				continue
			else:
				line_count+=1
				anim = row[0].strip()
				adj = row[1].strip()
				if anim == 'tiger' or adj in ['funny','happy','patient','reliable']:
					continue
				else:
					if anim in priors_anim:
						priors_anim[anim][adj] = (float(row[2])/100,float(row[3])/100)
					else:
						# print(row)
						priors_anim[anim] = {adj:(float(row[2])/100,float(row[3])/100)}
						animals.append(anim)

					if adj in priors_adj:
						priors_adj[adj][anim] = (float(row[2])/100,float(row[3])/100)
					else:
						priors_adj[adj] = {anim:(float(row[2])/100,float(row[3])/100)}
						adjectives.append(adj)

	# print(animals,adjectives)
	return priors_anim, priors_adj, animals, adjectives

#https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
def kl_divergence(p, q):
	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

class System:
	def __init__(self):
		self.priors_anim, self.priors_adj, self.animals, self.adjectives = priors_setup()
		self.lamb = 1
		self.norm_acceptability = self.normalized_p_f_num_f_val_given_c()
		self.norm_saliences = self.normalized_saliences()
		self.L0s = self.prepare_L0s()

	#how salient is it to talk about the meanness of sharks in the first place?
	def salience(self,cat,f_num):
		animal_mean, animal_sd = self.priors_anim[cat][f_num]
		x = np.arange(0, 1, 0.001)
		p_animal = norm.pdf(x,loc=animal_mean,scale=animal_sd)
		#for example...
		p_avg = norm.pdf(x, loc=0.5, scale=0.1)

		return kl_divergence(p_animal,p_avg)/len(x)

	#P(c, f_num)
	# 21 categories (incl. human) x 20 fts
	def normalized_saliences(self):
		norm_saliences = np.empty(shape=(len(self.animals), len(self.adjectives)))

		for c in self.animals:
			for ft in self.adjectives:
				salience = self.salience(c, ft)
				norm_saliences[self.animals.index(c),self.adjectives.index(ft)] = salience

		norm_saliences /= np.sum(norm_saliences)

		return norm_saliences

	#so for each category, there's going to be a prob distribution w/ 20 values and 20 fts. 
	def p_f_num_f_val_given_c(self, cat, f_num, f_val):
		epsilon = 0.01

		nonnorm = 1 - abs(self.priors_anim[cat][f_num][0] - f_val) - epsilon #or e

		return nonnorm

	#normalized acceptabilities for each category
	# {c : normalized_dict}
	def normalized_p_f_num_f_val_given_c(self):
		normalized_p_ft_val_given_c = {}

		for c in self.animals:
			p_for_this_c = {}
			for feature in self.adjectives:
				for value in [x[0] for x in self.priors_adj[feature].values()]:
					p_for_this_c[feature, value] = self.p_f_num_f_val_given_c(c, feature, value)

			p_for_this_c_normalized = {key : value/sum(p_for_this_c.values()) for key, value in p_for_this_c.items()}

			normalized_p_ft_val_given_c[c] = p_for_this_c_normalized

		return normalized_p_ft_val_given_c


	def L0(self, c, f_num, f_val, u):
		if c != u:
			return 0
		else:
			#P(f_num, f_val|c)
			#how acceptable is it to say "shark" to mean "0.9 scary"?
			# print(c)
			acceptability = self.norm_acceptability[c][(f_num, f_val)]

			#P(c, f_num)
			#how salient is it to talk about the meanness of sharks in the first place?
			salience = self.norm_saliences[self.animals.index(c), self.adjectives.index(f_num)]

			#need to renormalize!
			return acceptability * salience 

	def meta_L0(self, u):
		# {(c, f_num, f_val) : prob}
		L0_matrix = {}

		for c in ['human', u]:
			for feature in self.adjectives:
				for value in [x[0] for x in self.priors_adj[feature].values()]:
					L0_matrix[(c, feature, value)] = self.L0(c, feature, value, u)

		L0_matrix_renormalized = { key : value/sum(L0_matrix.values()) for key, value in L0_matrix.items()}

		return L0_matrix_renormalized

	def prepare_L0s(self):
		all_L0s = {}
		for u in self.animals:
			all_L0s[u] = self.meta_L0(u)

		return all_L0s

	def U(self, u, g, f_num, f_val):
		if g == f_num:
			if u != 'human':
				return math.log(self.L0s['human']['human', f_num, f_val] + self.L0s[u][u,f_num, f_val])
			else: 
				return math.log(self.L0s['human']['human', f_num, f_val])
		else:
			return 0

	def S1(self, u, g, f_val, f_num):

		all_Us = [math.e**(self.lamb * self.U(i, g, f_val, f_num)) for i in self.animals]

		#normalized
		return math.e**(self.lamb * self.U(u, g, f_val, f_num))/sum(all_Us)

	#condition = vague or specific
	def L1(self, c, f_num, f_val, u, condition):
		if c == 'human':
			p_c = 0.99
		else:
			p_c = 0.01

		p_f_num_f_val_given_c = self.norm_acceptability[c][f_num, f_val]

		if condition == 'na':
			p_g = 1/20
		else:
			if condition == f_num:
				#for instance; but fit is best when p_g is unform - explained in the paper
				# p_g = 0.5/20
				p_g = 1/20
			else:
				# p_g = 0.5/20
				p_g = 1/20

		g = f_num

		# there's a sum over g_s in the equation, but only the g term that's equal to (f_num, f_val) is going to get nonzero probability
		return p_c  * self.S1(u, g, f_num, f_val) * p_f_num_f_val_given_c

	def meta_l1(self, u, condition):
		# {(c, f_num, f_val) : prob}
		L1_matrix = {}

		for c in ['human', u]:
			for feature in self.adjectives:
				for value in [x[0] for x in self.priors_adj[feature].values()]:
					L1_matrix[(c, feature, value)] = self.L1(c, feature, value, u, condition)

		L1_matrix_renormalized = { key : value/sum(L1_matrix.values()) for key, value in L1_matrix.items()}

		return L1_matrix_renormalized

#######################################################
	'''
	given a prediction, return its probability

	e.g.
	utterance = bear
	context = 'na'
	prediction = ('strong', 0.9)

	corresponds to the scenario: "What is John like? He is a bear." 
	& inference that the speaker wanted to communicate 
	that John is 0.9 strong

	for each feature (e.g. strong), there are 20 degrees which can be inferred with nonzero probability.
	those are the degrees contained in priors_adj, [x[0] for x in self.priors_adj[feature].values()]
	'''

	def generate_prediction(self, utterance, context, predicted_ft, predicted_degree):
		L1_matrix = self.meta_l1(utterance, context)

		model_prob = L1_matrix[('human', predicted_ft, predicted_degree)]

		print(model_prob)

		return model_prob

def main():
	s = System()
	#probability that in the scenario: "What is John like? He is a lion" the listener will infer that John is as strong as is typical of a lion.
	s.generate_prediction('lion','na','strong',s.priors_anim['lion']['strong'][0])

main()
