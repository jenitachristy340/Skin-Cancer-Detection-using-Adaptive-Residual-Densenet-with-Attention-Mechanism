import time
import numpy as np


def update_positions(population, best_solution, lower_bound, upper_bound):
    new_population = np.copy(population)
    for i in range(population.shape[0]):
        r1 = np.random.rand()
        r2 = np.random.rand()
        inertia = r1 * (best_solution - population[i])
        social = r2 * (best_solution - population[i])
        new_position = population[i] + inertia + social
        new_position = np.clip(new_position, lower_bound, upper_bound)
        new_population[i] = new_position
    return new_population


# Waterwheel Plant Algorithm (WPA)
def WPA(population, fobj, VRmin, VRmax, max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    fitness = np.apply_along_axis(fobj, 1, population)

    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(max_iter):
        population = update_positions(population, best_solution, lb, ub)
        fitness = np.apply_along_axis(fobj, 1, population)

        current_best_fitness = np.min(fitness)
        current_best_solution = population[np.argmin(fitness)]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct



