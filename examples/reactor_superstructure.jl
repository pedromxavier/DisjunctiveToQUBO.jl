using DisjunctiveProgramming # master branch required 
using BARON

# Set parameters
cx = [1.5, 1]
cy = [6.1, 5.9]
α = [0.75, 0.8]
d = 3 # not specified in the problem, I think it is 3

# Create the model
model = GDPModel(BARON.Optimizer) # TODO replace with ToQUBO

# Add the variables
@variable(model, x[1:2] >= 0)
@variable(model, y[1:2], Logical)

# Add the objective
@objective(model, Min, cx' * x.^2 + cy' * binary_variable.(y))

# Create the disjunction constraint
@constraint(model, [i = 1:2], x[i] <= 5, Disjunct(y[i]))
@constraint(model, [(i, j) = [(1, 2), (2, 1)]], x[i] <= 0, Disjunct(y[j]))
disjunction(model, y, exactly1 = true) # explcitly show the keyword to add an exactly 1 constraint 

# Add the global constraint
@constraint(model, α'x >= d)

# Solve the model
optimize!(model, gdp_method = Indicator())

