import torch
import pickle
import math
import random
import os
from tqdm import tqdm
import copy
device='cuda:0'
count_sampler = 0
class DataSampler(object):
	   
	''' return DataSampler(datasetName=self.datasetName,mode='train',pos_dataset=self.train_set,whole_dataset=self.whole_dataset,
			batch_size=self.batch_size,entity_set=self.train_entity_set,relation_set=self.train_relation_set,
			neg_rate=self.neg_rate,groundtruth=self.groundtruth,possible_entities=self.possible_entities,rdrop=self.rdrop)
	'''
	def __init__(self,datasetName,mode,pos_dataset,whole_dataset,batch_size,entity_set,
	relation_set,neg_rate,rdrop=False,pos_neg_dataset=None):
		self.datasetName = datasetName

		self.mode = mode
		self.pos_dataset = pos_dataset
		self.whole_dataset = whole_dataset

		self.batch_size = batch_size
		self.entity_set = entity_set
		self.relation_set = relation_set

		self.neg_rate = neg_rate

		self.rdrop = rdrop

		if not os.path.exists('./sampler'):
			os.mkdir('./sampler')

		if mode=='train':
			global count_sampler
			count_sampler+=1
			dataset_path='sampler/{}-{}-{}-{}.pkl'.format(datasetName,mode,neg_rate,count_sampler)
		else:
			dataset_path='sampler/{}-{}-{}.pkl'.format(datasetName,mode,neg_rate)

		if os.path.exists(dataset_path):
			with open(dataset_path,'rb') as f:
				self.dataset = pickle.load(f)
		else:
			self.dataset = self.create_dataset(pos_dataset)
			with open(dataset_path,'wb') as fil:
				pickle.dump(self.dataset,fil)

		self.n_batch=math.ceil(len(self.dataset)/self.batch_size)
		self.i_batch=0

	def create_dataset(self,pos_dataset):
		"""
		Corrupt the head or tail of the given triplet
		"""
		dataset = [] 
		random.shuffle(pos_dataset)
		pos_dataset_set = set(pos_dataset)
		whole_dataset_set = set(self.whole_dataset)
		for triple in tqdm(pos_dataset):
			dataset.append((triple, 1))   
			h,r,t=triple
			for i in range(self.neg_rate):
				count=0
				while True:
					if(random.sample(range(2),1)[0]==1):
						#replace head
						replace_ent=random.sample(list(self.entity_set),1)[0]
						neg_triple=(replace_ent,r,t)
					else:
						#replace tail
						replace_ent=random.sample(list(self.entity_set),1)[0]
						neg_triple=(h,r,replace_ent)
					if neg_triple not in dataset:
						dataset.append((neg_triple,0))
						break
					else:
						dataset.append((neg_triple,1))
		return dataset

	def __iter__(self):
	  return self
	
	def __next__(self):
		if self.i_batch==self.n_batch:
			raise StopIteration()
		batch=self.dataset[self.i_batch*self.batch_size:(self.i_batch+1)*self.batch_size]
		if self.rdrop:
			batch=batch+batch
		self.i_batch+=1
		return batch

	def __len__(self):
		return self.n_batch
	
	def get_dataset_size(self):
		return len(self.dataset)

class DataLoader(object):
	def __init__(self,in_paths,batch_size,neg_rate=1,rdrop=False):
		self.datasetName = in_paths['dataset']
		self.data=self.datasetName
		#导入dataset
		self.train_set = self.load_dataset(in_paths['train'])
		if self.datasetName not in ['fb13']:
			self.valid_set = self.load_dataset(in_paths['valid'])
			self.test_set = self.load_dataset(in_paths['test'])
			self.valid_set_with_neg=None
			self.test_set_with_neg=None
		else:
			self.valid_set,self.valid_set_with_neg = self.load_dataset_with_neg(in_paths['valid'])
			self.test_set,self.test_set_with_neg = self.load_dataset_with_neg(in_paths['test'])

		self.whole_dataset=self.train_set+self.valid_set+self.test_set

		self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] + [t[-1] for t in (self.train_set + self.valid_set + self.test_set)])
		self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])

		self.batch_size=batch_size
		self.step_per_epc=math.ceil(len(self.train_set)*(1+neg_rate)/batch_size)    

		self.train_entity_set = set([t[0] for t in self.train_set] + [t[-1] for t in self.train_set])
		self.train_relation_set = set([t[1] for t in self.train_set])

		self.entity_list = sorted(self.entity_set)
		self.relation_list = sorted(self.relation_set)

		self.ent2id = {e:i for i,e in enumerate(sorted(self.entity_set))}
		self.rel2id = {r:i for i,r in enumerate(sorted(self.relation_set))}
		self.id2ent={i:e for i,e in enumerate(sorted(self.entity_set))}
		self.id2rel={i:r for i,r in enumerate(sorted(self.relation_set))}

		self.neg_rate=neg_rate
		self.rdrop=rdrop

	def load_dataset(self,in_path):
		"""
			加载dataset
			param:in_path 输入路径
			return:dataset
		"""
		dataset=[]
		with open(in_path,'r',encoding='utf-8') as fil:
			for line in fil.readlines():
				if in_path[-3:]=='txt':
					h,r,t = line.strip('\n').split('\t')
				else:
					h,r,t=line.strip('\n').split('\t')
				dataset.append((h,r,t))
		return dataset
	
	def load_dataset_with_neg(self,in_path):
		"""
			加载dataset
			param:in_path 输入路径
			return:dataset
		"""
		dataset=[]
		dataset_with_seg=[]
		with open(in_path,'r',encoding='utf-8') as fil:
			for line in fil.readlines():
				if in_path[-3:]=='txt':
					h,r,t,l = line.strip('\n').split('\t')
					
					if l=='-1':
						l=0
					else:
						l=1
					dataset.append((h,r,t))
					dataset_with_seg.append((h,r,t,l))
		return dataset,dataset_with_neg
	
	def train_data_sampler(self,batch_size,neg_rate):
		"""
			训练数据采样器
			param:batch_size batch大小
			return:DataSampler
		"""
		return DataSampler(datasetName=self.datasetName,mode='train',pos_dataset=self.train_set,whole_dataset=self.whole_dataset,
			batch_size=batch_size,entity_set=self.train_entity_set,relation_set=self.train_relation_set,neg_rate=neg_rate,rdrop=self.rdrop)

	def valid_data_sampler(self):
		"""
			验证数据采样器
			param:batch_size batch大小
			return:DataSampler
		"""
		return DataSampler(datasetName=self.datasetName,mode='valid',pos_dataset=self.valid_set,whole_dataset=self.whole_dataset,
			batch_size=self.batch_size,entity_set=self.entity_set,relation_set=self.relation_set,
			neg_rate=self.neg_rate,pos_neg_dataset=self.valid_set_with_neg)

	
	def test_data_sampler(self,batch_size,neg_rate):
		"""
			测试数据采样器
			param:batch_size batch大小
			return:DataSampler
		"""
		return DataSampler(datasetName=self.datasetName,mode='test',pos_dataset=self.test_set,whole_dataset=self.whole_dataset,
			batch_size=batch_size,entity_set=self.entity_set,relation_set=self.relation_set,neg_rate=neg_rate,pos_neg_dataset=self.test_set_with_neg)

	def get_dataset_size(self,split='train'):
		"""
			获取dataset大小
			param:split 划分
			return:dataset大小
		"""
		if split=='train':
			return len(self.train_set)*(1+self.neg_rate)
	
	def get_dataset(self,split):
		"""
			获取dataset
			param:split 划分
			return:dataset
		"""
		assert (split in ['train','valid','test'])
		if split == 'train':
			return self.train_set
		elif split == 'valid':
			return self.valid_set
		else:
			return self.test_set

	# def load_embedding(self):
	# 	"""
	# 	Load the pretrained embeddings
	# 	"""
	# 	if self.data=='wn18rr' or self.data=='wn18rr_cut':
	# 		triple2emb_file='/home/DYP/2023test/baseline/SimKGC-main/simkgc-emb-WN18RRtriple2emb-0.pkl'
	# 		n_batch=2713
	# 		# triple2emb_file='/home/DYP/2023test/baseline/LMKE_LSTM/LMKE_save_emb/LMKE-main/emb_data/wn18rr/wn18rr_triple2emb.pkl'
	# 	elif self.data=='fb15k-237':
	# 		triple2emb_file='/home/DYP/2023test/baseline/SimKGC-main/simkgc-emb-fb15k237triple2emb-10.pkl'
	# 		n_batch=8503
	# 		# triple2emb_file='/home/DYP/2023test/baseline/LMKE_LSTM/LMKE_save_emb/LMKE-main/emb_data/fb15k/fb15k237_triple2emb.pkl'
	# 	#triple2emb=pickle.load(open(triple2emb_file,'rb'))
	# 	triple2emb={}
	# 	with open(triple2emb_file,'rb') as f:
	# 		for i in range(n_batch):
	# 			i=pickle.load(f)
	# 			triple2emb.update(i)
	# 	return triple2emb   
	
	def load_embedding(self):
		"""
		Load the pretrained embeddings
		triple2emb={}
		"""
		if self.data=='wn18rr' or self.data=='wn18rr_cut':
			triple2emb_file='/home/DYP/2023test/baseline/SimKGC-main/data_emb/WN18RR_epc_emb.pkl'
		elif self.data=='fb15k-237':
			triple2emb_file='/home/DYP/2023test/baseline/SimKGC-main/data_emb/triple2emb.pkl'
		triple2emb=pickle.load(open(triple2emb_file,'rb'))
		return triple2emb

		# 找到具有相同头节点的子序列
	def find_subsequences(self,set):
		subgraph_size=4
		subgraphsequences = {}
		for triple in set:
			head = triple[0]
			if head not in subgraphsequences:
				subgraphsequences[head] = []
			if len(subgraphsequences[head]) <subgraph_size:
				subgraphsequences[head].append(triple)
		for k,v in subgraphsequences.items():
			while len(v) <subgraph_size:
				subgraphsequences[k].append((k,'0','0'))
		return subgraphsequences

	def load_subgraph_embedding(self,set,real_triples):
		subgraph_size=4
		triple2emb=self.load_embedding()
		head_subgraph_seqs=[]
		tail_subgraph_seqs=[]
		subgraphtriples = self.find_subsequences(set)

		for i, triple in enumerate(real_triples):
			h, r, t = triple
			if h in subgraphtriples.keys():
				head_subgraph_triples=subgraphtriples[h]
			else:
				head_subgraph_triples=[]
				while len(head_subgraph_triples)<subgraph_size:
					head_subgraph_triples.append((h,'0','0'))
			if t in subgraphtriples.keys():
				tail_subgraph_triples=subgraphtriples[t]
			else:
				tail_subgraph_triples=[]
				while len(tail_subgraph_triples)<subgraph_size:
					tail_subgraph_triples.append((h,'0','0'))
			head_subgraphseq = []
			tail_subgraphseq=[]
			for triple in head_subgraph_triples:
				if triple in triple2emb.keys():
					head_subgraphseq.append(triple2emb[triple].to(device))
				else:
					head_subgraphseq.append(torch.zeros(1536).to(device))
			head_subgraph_seq=torch.stack(head_subgraphseq,dim=0).to(device)
			head_subgraph_seqs.append(head_subgraph_seq)									
			for triple in tail_subgraph_triples:
				if triple in triple2emb.keys():
					tail_subgraphseq.append(triple2emb[triple].to(device))
				else:
					tail_subgraphseq.append(torch.zeros(1536).to(device))
			tail_subgraph_seq=torch.stack(tail_subgraphseq,dim=0).to(device)
			tail_subgraph_seqs.append(tail_subgraph_seq)

		head_seq=torch.stack(head_subgraph_seqs,dim=0).to(device)
		tail_seq=torch.stack(tail_subgraph_seqs, dim=0).to(device).squeeze()
		return head_seq,tail_seq
	
def random_choose(random_ratio, constrain_ratio, reverse_ratio):
	x = random.random()
	if x <= random_ratio:
		return 'random'
	elif x <= (random_ratio + constrain_ratio):
		return 'constrain'
	else:
		return 'reverse'