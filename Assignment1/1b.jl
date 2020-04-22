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
    score = 1 - sum(((Y - mean_vec)).^2)/(sum(((X * C) - Y).^2))
    return score
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

# checking the performance of the model
function performance(X_test, Y_test, newB)
  score = rscore(X_test, Y_test, newB);
  rerror  = rmse(X_test, Y_test, newB);
  return score, rerror
end


# gradient descent
function gradientDescent(X_train, Y_train, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    error = zeros(numIterations)
    score = zeros(numIterations)


    m = length(Y-train)
  
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
# bathrooms = dataset.bathrooms
sqft_living = dataset.sqft_living
bed_2 = bedrooms.*bedrooms
sqft_2 = sqft_living.*sqft_living
bed_sqft = bedrooms.*sqft_living
Y = dataset.price

bedrooms = norm(bedrooms)
# bathrooms = norm(bathrooms)
sqft_living = norm(sqft_living)
bed_2 = norm(bed_2)
sqft_2 = norm(sqft_2)
bed_sqft = norm(bed_sqft)

m = length(Y)

# splitting dataset into training, validation and test set

bedrooms_train = bedrooms[1:17290,:]
bedrooms_test = bedrooms[17290:m,:]

sqft_living_train = sqft_living[1:17290,:]
sqft_living_test = sqft_living[17290:m,:]

bed_2_train = bed_2[1:17290,:]
bed_2_test = bed_2[17290:m,:]

sqft_2_train = sqft_2[1:17290,:]
sqft_2_test = sqft_2[17290:m,:]

bed_sqft_train = bed_sqft[1:17290,:]
bed_sqft_test = bed_sqft[17290:m,:]

Y_train = Y[1:17290,:]
Y_test = Y[17290:m , :];

x0 = ones(m);
X_test = cat(x0[17290:m],bedrooms_test , sqft_living_test , bed_2_test, sqft_2_test, bed_sqft_test, dims = 2);
X_train = cat(x0[1:17290],bedrooms_train , sqft_living_train , bed_2_train, sqft_2_train, bed_sqft_train, dims = 2);

B = zeros(6,1)

# performing gradient descent
learning_rate = 0.001;
no_iter = 10000;

newB, costHistory, error, score = gradientDescent(X_test, Y_test, B, learning_rate, no_iter);

# CSV handling
pred = X*newB
pre=zeros(m,1) #temporary variable for storing predicted prices
for i in 1:m
pre[i]=pred[i]
end

df = DataFrame(Actual = Y, Predicted = pred[:])
CSV.write("data\\1b.csv",df)

#final output
r_score,r_mse = performance(X_test,Y_test,newB)
r_score, r_mse, newB