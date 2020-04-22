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

# cost function with the penalty
function costFunction(X, Y, C,lambda)
    sumB=zeros(4,1)
    for i in 1:4
    sumB[i]=abs(B[i])
    end

    m = length(Y)
    cost = sum(((X * C) - Y).^2)/(2*m) + lambda*sum(sumB.^2)/4
    return cost
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


# splitting dataset into training, validation and test set

dataset = CSV.read("house.csv")

bedrooms = dataset.bedrooms
bathrooms = dataset.bathrooms
sqft_living = dataset.sqft_living
Y = dataset.price

bedrooms = norm(bedrooms)
bathrooms = norm(bathrooms)
sqft_living = norm(sqft_living)

m = length(Y)

x0 = ones(m);

bedrooms_train = bedrooms[1:12966,:]
bedrooms_test = bedrooms[12967:17289,:]
bedrooms_val = bedrooms[17290:m,:]

sqft_living_train = sqft_living[1:12966,:]
sqft_living_test = sqft_living[12967:17289,:]
sqft_living_val = sqft_living[17290:m,:]

bathrooms_train = bathrooms[1:12966,:]
bathrooms_test = bathrooms[12967:17289,:]
bathrooms_val = bathrooms[17290:m,:]


Y_train = Y[1:12966,:]
Y_test = Y[12967:17289,:];
Y_val = Y[17290:m,:];

X_test = cat(x0[12967:17289],bedrooms_test , sqft_living_test ,bathrooms_test, dims = 2);
X_train = cat(x0[1:12966],bedrooms_train , sqft_living_train ,bathrooms_train, dims = 2);
X_val = cat(x0[17290:m],bedrooms_val , sqft_living_val , bathrooms_val, dims = 2);



# performing gradient descent
function gradientDescent(X, Y, B, learningRate, numIterations,lambda)
    costHistory = zeros(numIterations)
    error = zeros(numIterations)
    score = zeros(numIterations)


    m = length(Y)
  
    for iteration in 1:numIterations
        
        loss = (X * B) - Y
        gradient = (X' * loss)/m + (lambda/2)*B;
        
        B = B - (learningRate * gradient)

        m = length(Y)
        er = rmse(X,Y,B)
        
      # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B,alfa)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
        
        #Extra paremeters
        error[iteration] = er
        score[iteration] = rscore(X, Y, B)
    end
    return B, costHistory, error, score
end

B = zeros(4,1)

#setting hyperparameters
learning_rate = 0.0001;
no_iter = 10000;

#Finding the optimum value
index=0 
alfa=-10
error=zeros(20,1)
lambda=zeros(20,1)
min_er=5000000

for i in 1:20
lambdaa[i]=alfa
newB, _,_, _ = gradientDescent(X_val, Y_val, B, 0.0001, 10000,lambda[index])
error[i]=rmse(X_test, Y_test, newB)

min_er=min(error[i],min_er)
if min_er==error[i]
index=i
end

alfa+=1
end

print(lambdaa[index]) #optimum lambda

#checking the performance of the model
newB, costHistory, error, score = gradientDescent(X_train, Y_train, B, learning_rate, no_iter,lambda[index])

rscore_val, rmse_val = performance(X_test,Y_test,newB)

#CSV handling
pred = X_test*newB
pre=zeros(length(Y_test),1) #temporary variable for storing predicted prices
for i in 1:length(Y_test)
pre[i]=pred[i]
end

df = DataFrame(Predicted = pred[:])
CSV.write("data\\2a.csv",df)

#final output
rscore_final, rmse_final = performance(X_test,Y_test,newB)
rscore_final, rmse_final, newB