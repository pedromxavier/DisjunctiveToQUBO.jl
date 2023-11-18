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
@variable(model, 0 <= x[1:2] <= 5)
@variable(model, y[1:2], Logical)

# Add the objective
@objective(model, Min, cx' * x.^2 + cy' * binary_variable.(y))

# Create the disjunction constraint
@constraint(model, x[2] <= 0, Disjunct(y[1]))
@constraint(model, x[1] <= 0, Disjunct(y[2]))
disjunction(model, y, exactly1 = true) # explcitly show the keyword to add an exactly 1 constraint 

# Add the global constraint
@constraint(model, α'x >= d)

# Solve the model
optimize!(model, gdp_method = Indicator())

