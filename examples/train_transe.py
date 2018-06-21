from openke import Dataset
from openke.models import TransE

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
model = lambda: TransE(50, 1.0, base.shape, batchshape=(len(base) // 20, 2))

#   Train the model.
base.train(model, folds=20, epochs=500, batchkwargs={'negatives':(1,0), 'bern':False, 'workers':4},
	eachepoch=print)

#   Save the result.
model.save("./result")
