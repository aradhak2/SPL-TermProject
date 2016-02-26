Pkg.add("Images")
Pkg.add("DataFrames")

using Images
using DataFrames
using ColorTypes
using Iterators

function read_data(typeData, labelsInfo, imageSize, path)
	x = zeros(size(labelsInfo, 1), imageSize)

	for (index, idImage) in enumerate(labelsInfo[:id])
		nameFile = joinpath(homedir(), "Julia", "$(typeData)Resized/$(idImage).bmp")
		#println(nameFile)
    img = load(nameFile)

		temp = float32(img)
		if ndims(temp) == 3
			temp = mean(temp.data, 1)
		end

	  x = [iter::RGB4{Float32} for iter in temp]
  end
  return x
end

imageSize = 400

path = joinpath(homedir(), "Julia")

labelsInfoTrain = readtable("$(path)/Data/trainLabels_20.csv")
xTrain = read_data("train", labelsInfoTrain, imageSize, path)
xTrain = [iter::RGB4{Float32} for iter in xTrain]
println(typeof(xTrain))

labelsInfoTest = readtable("$(path)/Data/testLabels_20.csv")
xTest = read_data("test", labelsInfoTest, imageSize, path)

yTrain = map(x -> x[1], labelsInfoTrain[:id])
yTrain = [iter::Integer for iter in yTrain]
println(typeof(yTrain))


Pkg.add("DecisionTree")
using DecisionTree

model = build_forest(yTrain, xTrain, convert(Integer, 20), convert(Integer, 50), 1.0)
println(model)
println(typeof(model))
