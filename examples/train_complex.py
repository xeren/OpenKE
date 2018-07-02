from openke import Dataset
from openke.models import ComplEx as Model

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
folds, negatives = 20, (1,0)
model = lambda: Model(50, .0001, base.shape,
	batchshape=(len(base)//folds, 1+sum(negatives)))

#   Train the model.
model, record = base.train(model, folds=folds, epochs=50,
	batchkwargs={'negatives':negatives, 'bern':False, 'workers':4},
	eachepoch=print, prefix="./result.complex")
print(record)

#   Input testing files from benchmarks/FB15K/.
test = Dataset("./benchmarks/FB15K/test2id.txt")

#   Perform a test.
print(test.meanrank(model, head=False, label=False))
