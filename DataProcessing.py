import csv
import numpy as np
import Constants as const
import math
from random import randint
import matplotlib.pyplot as plt

# Our data looks like this:
#
#     User | Movie | Rating
#     -----|-------|-------
#       44 |     1 |      4
#       61 |     1 |      3
#       67 |     1 |      4
#       72 |     1 |      3
#       86 |     1 |      5
#
userIndex = 0
movieIndex = 1
ratingIndex = 2
# We add a fourth column containing the rating-to-average offset
offsetIndex = 3

def CalculateBetterMean(data, overallMean, overallVar):
	"""Just taking the mean as `sum/len` gives us very bad approximations for the
	mean if there is little data. Eg, if a movie is only rated once, as 1/5, it is
	unlikely that the true mean is 1.0.
	"""
	localMean = np.average(data)
	localSum = np.sum(data)
	localVar = np.var(data)
	localSize = len(data)
	ratio = localVar / overallVar # In paper is used a const of 25
	betterMean = (overallMean * ratio + localSum) / (ratio + localSize)
	return betterMean

# The logic of this is entirely following this blog: http://sifter.org/~simon/journal/20061211.html
def PredictBaseline(train):
	"""Find the (better) rating mean of each movie, and the (better) user rating
	as well as the mean of the rating offsets for each user.
	"""
	# Mean of all ratings
	overallMean = np.average(train[:,ratingIndex])
	# Variance of all ratings
	overallVar = np.var(train[:,ratingIndex])
	# Mean rating for each movie
	averageRate_Movies = []
	for movie in range(1, const.Number_Of_Movies + 1):
		# Select all rows in the table for this movie
		ratings = train[np.equal(train[:,movieIndex], movie)][:,ratingIndex]
		betterMean = CalculateBetterMean(ratings, overallMean, overallVar)
		averageRate_Movies.append(betterMean)
	# For each rating, calculate the offset from the average rating for that movie to the actual rating
	(height, width) = train.shape
	offsetData = np.zeros((height, width + 1))
	offsetData[:, :-1] = train
	for item in offsetData:
		item[offsetIndex] = item[ratingIndex] - averageRate_Movies[int(item[movieIndex]) - 1]
	# Mean offset from movie rating for each user:
	# Find average movie rating. For each movie, we expect a user to rate
	# it the average. If the rating differ, there is an offset. Average this offset.
	averageOffset_Users = []
	for user in range(1, const.Number_Of_Users + 1):
		offsets = offsetData[np.equal(train[:,userIndex], user)][:,offsetIndex]
		betterMean = CalculateBetterMean(offsets, overallMean, overallVar)
		averageOffset_Users.append(betterMean)
	return {
		"AverageRate_Movies": averageRate_Movies,
		"AverageOffset_Users": averageOffset_Users #, "OffsetData":offsetData
	}

def GetBaselinePrediction(predictions, movie, user):
	averageRate_Movies = predictions["AverageRate_Movies"]
	averageOffset_Users = predictions["AverageOffset_Users"]
	prediction = averageRate_Movies[movie - 1] + averageOffset_Users[user - 1]
	prediction = max(const.Minimum_Rating, prediction)
	prediction = min(const.Maximum_Rating, prediction)
	return prediction

def OutputPredictions(predictions, outputFile):
	np.savetxt(outputFile, predictions,
	delimiter=",", fmt="%s,%s", header="ID,Prediction", comments="")

def OutputProperFormatData(data, outputFile):
	np.savetxt(outputFile, data,
	delimiter=",", fmt="%s,%s,%s", header="Movie,User,Prediction", comments="")

def GetProcessedData(file):
	data = []
	with open(file, 'r') as csvfile:
		data_raw = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(data_raw)
		for item in data_raw:
			data.append([int(item[0]), int(item[1]), int(item[2])])
	data = np.array(data)
	return data

# Output SVD features after training so no preprocessing is necessary
def OutputSVDFeatures(baselinePredictions, userValues, movieValues, userBias, movieBias):
	averageRate_Movies = baselinePredictions["AverageRate_Movies"]
	averageOffset_Users = baselinePredictions["AverageOffset_Users"]
	#
	np.savetxt("BaselinePrediction_AverageRate_Movie.csv", averageRate_Movies,
	delimiter=",", fmt="%f", header="AverageRate", comments="")
	np.savetxt("BaselinePrediction_AverageOffset_User.csv", averageOffset_Users,
	delimiter=",", fmt="%f", header="AverageOffset", comments="")
	#
	np.savetxt("SVD_UserFeatures.csv", userValues, delimiter=",")
	np.savetxt("SVD_FeaturesMovies.csv", movieValues, delimiter=",")
	np.savetxt("SVD_MoviesBias.csv", movieBias, delimiter=",")
	np.savetxt("SVD_UserBias.csv", userBias, delimiter=",")

def predictRating(baselinePredictions, userValues, movieValues, u, m, movie, user, useBaseline):
	if(useBaseline):
		baselinePrediction = GetBaselinePrediction(baselinePredictions, movie, user)
		return baselinePrediction + sum(userValues[user] * movieValues[:,movie]) + u[user] + m[movie]
	else :
		return sum(userValues[user] * movieValues[:,movie]) + u[user] + m[movie]

#  (root mean squared error)
def RMSE (baselinePredictions, userValues, movieValues, u, m, controlSample, useBaseline):
	results = []
	for sample in controlSample:
		prediction = predictRating(baselinePredictions, userValues, movieValues, u, m,
			sample[movieIndex]-1, sample[userIndex]-1, useBaseline)
		err = sample[ratingIndex] - prediction;
		results.append(err*err)
	return math.sqrt(np.mean(results))

def GetPredictions(baselinePredictions, userValues, movieValues, u, m, test):
	predictions = []
	for value in test:
		prediction = predictRating(baselinePredictions, userValues, movieValues, u, m, value[movieIndex]-1, value[userIndex]-1, True)
		prediction = max(const.Minimum_Rating, prediction)
		prediction = min(const.Maximum_Rating, prediction)
		predictions.append(["r%d_c%d" % (value[0], value[1]), prediction])
	return predictions

def SplitTrainData(train):
	trainSize = len(train)
	controlSampleSize = int(trainSize/const.Sample_Size_Percentage)
	controlSampleIndexes = np.random.choice(range(0,trainSize,1), size=controlSampleSize, replace=False, p=None)
	controlSampleIndexes.sort()
	trainSampleIndexes = np.delete(range(0,len(train),1), controlSampleIndexes)
	# Select new train and sample data
	controlSample = train[controlSampleIndexes]
	trainSample = train[trainSampleIndexes]
	return {
		"ControlSample": controlSample, 
		"TrainSample": trainSample
	}

print ("Reading input data.")
train = GetProcessedData(const.Processed_Train_Data_Location)
test = GetProcessedData(const.Processed_Test_Data_Location)

#baselinePredictions = PredictBaseline(trainSample)
print ("Predicting baseline.")
baselinePredictions = PredictBaseline(train)
baselineMovies = np.array(baselinePredictions["AverageRate_Movies"])
baselineUsers = np.array(baselinePredictions["AverageOffset_Users"])

print ("Split data into test and train.")
splitData = SplitTrainData(train)
controlSample = splitData["ControlSample"]
trainSample = splitData["TrainSample"]

#
# We are using random 90% for training, so the training results might vary a bit.
#

numberOfFeatures = 90
lrate = 0.001
K = 0.02
B = 0.05
# our mean is 3.8572805008190647 which means it could be better to try out different things
globalMean = 3.6033 

u = np.zeros(const.Number_Of_Users) 
m = np.zeros(const.Number_Of_Movies)

userValues = np.zeros((const.Number_Of_Users, numberOfFeatures))
movieValues = np.zeros((numberOfFeatures, const.Number_Of_Movies))

curErrorRate = RMSE(baselinePredictions, userValues, movieValues, u, m, controlSample, True);
newErrorRate = curErrorRate - 0.01;
print("Starting training...")
for feature in range(0, numberOfFeatures):
	print("Feature %d" % feature)
	userValues[:,feature] = 0.1
	movieValues[feature] = 0.1
	if(feature % 10 == 0 and feature > 1):
		OutputSVDFeatures(baselinePredictions, userValues, movieValues, u, m)
	for i in range(0, 1000):
		print ("%0.10f %0.10f %0.10f" % (curErrorRate, newErrorRate, curErrorRate - newErrorRate))
		if(curErrorRate < newErrorRate and i > 3):
			curErrorRate = newErrorRate
			newErrorRate = curErrorRate - 0.1
			break;
		curErrorRate = newErrorRate
		for item in trainSample:
			user = item[userIndex] - 1; # Account for indexing
			movie = item[movieIndex] - 1; # Account for indexing
			mv = movieValues[:, movie][feature]
			uv = userValues[user][feature]
			#
			rating = item[ratingIndex]
			#
			prediction = predictRating(baselinePredictions, userValues, movieValues, 
				u, m, movie, user, True)
			#
			err = rating - prediction
			biasUpdate = lrate * (err - B *(u[user] + m[movie] - globalMean))
			#
			u[user] += biasUpdate
			m[movie] += biasUpdate
			userValues[user][feature] += lrate * (err * mv - K * uv);
			movieValues[:,movie][feature] += lrate * (err * uv - K * mv);
		#
		newErrorRate = RMSE(baselinePredictions, userValues, movieValues, u, m, controlSample, True);	

finalPredictions = GetPredictions(baselinePredictions, userValues, movieValues, u, m, test)
#plt.hist(x, bins='auto')
#plt.show()
OutputPredictions(finalPredictions, const.Prediction_Data_Location)
OutputSVDFeatures(baselinePredictions, userValues, movieValues, u, m)
		
errorRate = RMSE(baselinePredictions, userValues, movieValues, u, m, train, True);
baselineErrorRate = RMSE(baselinePredictions, userValues, movieValues, u, m, controlSample, True);
print("********")
print("Error rate control %f "  % errorRate)
print("Baseline error rate control %f " % baselineErrorRate)


finalPredictions = GetPredictions(baselinePredictions, userValues, movieValues, u, m, test)
#plt.hist(x, bins='auto')
#plt.show()
OutputPredictions(finalPredictions, const.Prediction_Data_Location)


