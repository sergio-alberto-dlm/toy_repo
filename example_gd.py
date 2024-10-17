import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X_b, y, theta):
    m = len(X_b)
    predictions = X_b.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.sum(np.square(errors))

def gradient_descent(X_b, y, learning_rate, n_iterations):
    theta = np.random.randn(2, 1) 
    cost_history = [] 
    m = len(X_b)

    for iteration in range(n_iterations):
        gradients = 1/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients

        # Compute the cost at each iteration
        cost = compute_cost(X_b, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def plot_cost_over_iterations(cost_history, n_iterations):
    plt.figure(figsize=(8, 5))
    plt.plot(range(n_iterations), cost_history, 'b-', label="Cost")
    plt.title('Cost over Iterations (Gradient Descent)')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X = 2 * np.random.rand(100, 1)
    y = 4*np.cos(X) + 3 * np.sin(X)*np.sin(X) + X**2 + np.random.randn(100, 1)

    # Add a bias term (x0 = 1) to each instance
    X_b = np.c_[np.ones((100, 1)), X]

    # Gradient Descent parameters
    learning_rate = 0.01
    n_iterations = 1000
    m = len(X_b)

    # Perform gradient descent
    theta, cost_history = gradient_descent(X_b, y, learning_rate, n_iterations)
    
    # Plot the cost over iterations
    plot_cost_over_iterations(cost_history, n_iterations)

    # Final theta values
    print(f"Final parameters: theta_0 = {theta[0][0]}, theta_1 = {theta[1][0]}")

# Entry point
if __name__ == "__main__":
    main()
