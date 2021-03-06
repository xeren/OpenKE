#coding:utf-8
class Dataset(object):
	'''
		Manages a collection of relational data
		encoded as a set of triples, each describing a statement (or fact)
		over two objects called 'entities', one 'head' and one 'tail',
		being related in a manner that is symbolized by a relation 'label'.

		The application encodes both entities and relations as integral values,
		describing an index in an ordered table.
	'''


	def __init__(self, filename,
			entities, relations,
			library='./libopenke.so'):
		'''
			Creates a new dataset from a table in a file.

				Arguments
			filename - String identifying a file to import.
			entities - Upper bound of entity indexes.
			relations - Upper bound of relation indexes.
			library - Path to a shared object implementing the preprocessing.
			See this project's `base/openke.hpp` for its interface.

				Note
			At the moment, no two datasets can be open at the same time!
			The file is expected to contain whitespace-separated integral values,
			where the first value denotes the 'amount' of statements.
			The remainder should be a sequence of 'amount' times three indexes
			in order 'head', 'tail', 'label'.
		'''

		global _l
		from ctypes import c_char_p
		self.__library = _l[library]

		self.__library.importTrainFiles(c_char_p(bytes(filename, 'utf-8')),
				entities, relations)
		self.size = self.__library.getTrainTotal()
		self.shape = entities, relations


	def __len__(self):
		return self.size


	def batch(self, count, negatives=(0,0), bern=True, workers=1, seed=1):
		'''
			Separates the dataset into nearly equal parts.
			Iterates over all parts, each time yielding four arrays.
			This method can be used for cross-validation as shown below:

				assert numpy, base, folds, epochs, Model

				for i in range(folds):

					model = Model()

					for _ in range(epochs):
						for j, b in enumerate(base.batch(folds)):
							if i != j:
								model.fit(*b)

					for j, b in enumerate(base.batch(folds)):
						if i == j:
							break
					score = model.predict(*b[:3])
		'''

		from numpy import float32, int64, zeros

		size = self.size // count
		S = size * (1 + sum(negatives[:2]))
		types = [int64, int64, int64, float32]
		batch = [zeros(S, dtype=t) for t in types]
		h, t, l, y = [_carray(x) for x in batch]

		self.__library.randReset(workers, seed)

		sampling = self.__library.bernSampling if bern else self.__library.sampling
		for _ in range(count):
			sampling(h, t, l, y, size, negatives[0], negatives[1], workers)
			yield batch


	def train(self, model,
			folds=1, epochs=1,
			batchkwargs={},
			prefix='best',
			eachepoch=None, eachbatch=None):
		'''
			A simple training algorithm over the whole set.

				Arguments
			model - Parameterless constructor of to-be-trained embedding models.
			epochs - Integral amount of repeated training epochs
			folds - Integral amount of batches t
			bern - Truth value whether or not to use Bernoille distribution
			when choosing head or tail to be corrupted.
			workers - Integral amount of worker threads for generation.
			seed - Seed for the random number generator.
			eachepoch - Callback at the end of each epoch.
			eachbatch - Callback after each batch.
		'''

		record = list()
		for index in range(folds):

			m = model()
			for epoch in range(epochs):

				loss = 0
				for i, batch in enumerate(
						self.batch(folds, **batchkwargs)):
					if i == index:
						continue

					# one training batch
					loss += m.fit(*batch)

					eachbatch and eachbatch(batch, loss)

				eachepoch and eachepoch(epoch, loss)

			# choose the best-performing model
			record.append(self.meanrank(m, folds, index))
			if min(record, key=sum) == record[-1]:
				m.save(prefix)

		m.restore(prefix)
		return m, record


	def meanrank(self, model, folds=1, index=0,
			head=True, tail=True, label=True, batchkwargs={}):
		'''
			Computes the mean rank of link prediction of one batch of the data.
			Returns floating values between 0 and size-1
			where lower results denote better models.
			The return value consists of up to three values,
			one for each of the columns 'head', 'tail' and 'label'.

				Arguments
			model - The to-be-tested embedding model.
			folds - Amount of batches the data is separated.
			index - Identifier of the tested batch.
			head, tail, label - Truth values denoting
			whether or not the respecting column should be tested.

				Note
			This test filters only 'false' facts evaluating better than the question.
			See `openke.meanrank` for the unfiltered, or 'raw', version.
		'''

		def rank(d, x, h, t, l):
			y, z = self.query(h, t, l), model.predict(h, t, l)
			return sum(1 for i in range(self.shape[d]) if z[i] < z[x] and not y[i])

		I = lambda: range(self.size // folds)
		for i, (h, t, l, _) in enumerate(self.batch(folds, **batchkwargs)):
			if i == index:
				break
		ranks = [
				(rank(0, h[i], None, t[i], l[i]) for i in I()) if head else None,
				(rank(0, t[i], h[i], None, l[i]) for i in I()) if tail else None,
				(rank(1, l[i], h[i], t[i], None) for i in I()) if label else None]

		return [sum(i) / self.size for i in ranks if i is not None]


	def query(self, head, tail, relation):
		'''
			Checks which facts are stored in the entire dataset.
			This method is overloaded for the task of link prediction,
			awaiting an incomplete statement and returning all known substitutes.

				Arguments
			head - Index of a head entity.
			tail - Index of a tail entity.
			label - Index of a relation label.

				Return Value
			A boolean array, deciding for each candidate
			whether or not the resulting statement is contained in the dataset.
		'''
		from numpy import bool_, zeros
		if head is None:
			if tail is None:
				if relation is None:
					raise NotImplementedError('querying everything')
				raise NotImplementedError('querying full relation')
			if relation is None:
				raise NotImplementedError('querying full head')
			heads = zeros(self.shape[0], bool_)
			self.__library.query_head(_carray(heads), tail, relation)
			return heads
		if tail is None:
			if relation is None:
				raise NotImplementedError('querying full tail')
			tails = zeros(self.shape[0], bool_)
			self.__library.query_tail(head, _carray(tails), relation)
			return tails
		if relation is None:
			relations = zeros(self.shape[1], bool_)
			self.__library.query_rel(head, tail, _carray(relations))
			return relations
		raise NotImplementedError('querying single facts')
# FIXME backwards-compatibility
Config = Dataset


def meanrank(h, t, l, model, head=True, tail=True, label=True):
	'''
		Computes the mean rank of link prediction of its entire content.

			Arguments
		h, t, l - Integral arrays of equal shape describing relational questions.
		model - The to-be-tested embedding model.
		head, tail, label - Truth values, each of which denoting
		whether or not the respecting column should be tested.

			Return Value
		Flaoting point values between 0 and size-1
		where lower results denote better models.
		The result consists of up to three values,
		one for each of the columns 'head', 'tail' and 'label' in this order.

			Note
		This test ignores whether or not statements
		that evaluate better than the question are known truths or not.
		See `openke.Database.meanrank` for the filtered version.
	'''

	def rank(x, h, t, l):
		z = model.predict(h, t, l)
		return sum(1 for i in range(self.shape[0]) if z[i] < z[x])

	I = lambda: range(len(h))
	ranks = [
			(rank(h[i], None, t[i], l[i]) for i in I()) if head else None,
			(rank(t[i], h[i], None, l[i]) for i in I()) if tail else None,
			(rank(l[i], h[i], t[i], None) for i in I()) if label else None]

	return [sum(i) / self.size for i in ranks if i is not None]


def _carray(a):
	return a.__array_interface__['data'][0]


class _Library:
	'''
		Manages the connection to the library.
	'''


	def __init__(self):
		self.__dict = dict()


	def __getitem__(self, key):
		if key in self.__dict:
			return self.__dict[key]
		from ctypes import c_void_p as p, c_int64 as i, cdll
		l = cdll.LoadLibrary(key)
		l.sampling.argtypes = [p, p, p, p, i, i, i, i]
		l.bernSampling.argtypes = l.sampling.argtypes
		l.query_head.argtypes = [p, i, i]
		l.query_tail.argtypes = [i, p, i]
		l.query_rel.argtypes = [i, i, p]
		l.importTrainFiles.argtypes = [p, i, i]
		l.randReset.argtypes = [i, i]
		self.__dict[key] = l
		return l


_l = _Library()
