# importing libraries
import Pkg

Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("DataFrames")

# installing packages
using CSV
using Plots
using DataFrames

# Extra parameters : RMSE , R2score

function rmse(X, Y, C)
    m = length(Y)
    cer = (sum(((X * C) - Y).^2)/(m))^(1/2)
    return cer
end

function rscore(X, Y, C)
    m = length(Y)
    mean = sum(Y)/m
    mean_vec = mean*ones(m,1)
    score = 1 - (sum(((X * C) - Y).^2))/sum(((Y - mean_vec).^2))
    return score
end

#performance checking

function performance(X_test, Y_test, newB)
  score = rscore(X_test, Y_test, newB);
  rerror  = rmse(X_test, Y_test, newB);
  return score, rerror
end

# Data normalization
function norm(Y)
  m = length(Y);
  mean = sum(Y)/m;
  mean_vec = mean*ones(m,1)
  var = sum((Y.^2)/m) - (mean.^2);
  x = (Y-mean_vec)/sqrt(var);
  return x
end


# cost function
function costFunction(X, Y, C)
    m = length(Y)
    cost = sum(((X * C) - Y).^2)/(2*m)
    return cost
end

# gradient descent
function gradientDescent(X_train, Y_train, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    error = zeros(numIterations)
    score = zeros(numIterations)


    m = length(Y_train)
  
    for iteration in 1:numIterations
        
        loss = (X_train * B)-Y_train

        
        gradient = (X_train' * loss)/m
        
        B = B - learningRate * gradient

        m = length(Y_train)
        er = rmse(X_train,Y_train,B)
        
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X_train, Y_train, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
        
        #Extra paremeters
        error[iteration] = er
        score[iteration] = rscore(X_train, Y_train, B)
    end
    return B, costHistory, error, score
end

# loading and normalizing dataset
dataset = CSV.read("house.csv")

bedrooms = dataset.bedrooms
bathrooms = dataset.bathrooms
sqft_living = dataset.sqft_living
Y = dataset.price

m = length(Y)

bedrooms = norm(bedrooms)
bathrooms = norm(bathrooms)
sqft_living = norm(sqft_living)

x0 = ones(m);

bedrooms_train = bedrooms[1:17289,:]
bedrooms_test = bedrooms[17290:m,:]

sqft_living_train = sqft_living[1:17289,:]
sqft_living_test = sqft_living[17290:m,:]

bathrooms_train = bathrooms[1:17289,:]
bathrooms_test = bathrooms[17290:m,:]

X_test = cat(x0[17290:m],bedrooms_test , bathrooms_test , sqft_living_test, dims = 2);
X_train = cat(x0[1:17289],bedrooms_train , bathrooms_train , sqft_living_train, dims = 2);

Y_train = Y[1:17289]
Y_test = Y[17290:m];

B = zeros(4,1)

# performing gradient descent
learning_rate = 0.001;
no_iter = 20000;

newB, costHistory, error, score = gradientDescent(X_test, Y_test, B, learning_rate, no_iter);

# CSV handling
pred = X*newB
pre=zeros(m,1) #temporary variable for storing predicted prices
for i in 1:m
pre[i]=pred[i]
end

df = DataFrame(Actual = Y_test, Predicted = pred[:])
CSV.write("data\\1a.csv",df)

#final output
r_score,r_mse = performance(X_test,Y_test,newB)
r_score, r_mse, newB
