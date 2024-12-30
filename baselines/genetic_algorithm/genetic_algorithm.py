import torch
from typing import Callable, Tuple


class GeneticAlgorithm:
    def __init__(
            self,
            vector_length: int,
            population_size: int,
            generations: int,
            mutation_stddev: float,
            fitness_function: Callable[[torch.Tensor], torch.Tensor],
            device: str = None,
    ) -> None:
        """
        Initialize the Genetic Algorithm.

        Parameters:
        - vector_length (int): Length of each vector (individual).
        - population_size (int): Number of individuals in the population.
        - generations (int): Number of generations to evolve.
        - mutation_stddev (float): Standard deviation for mutation noise.
        - fitness_function (Callable): A user-defined function to evaluate fitness.
        - device (str): 'cuda' for GPU or 'cpu' for CPU. Defaults to auto-detection.
        """
        self.vector_length: int = vector_length
        self.population_size: int = population_size
        self.generations: int = generations
        self.mutation_stddev: float = mutation_stddev
        self.fitness_function: Callable[[torch.Tensor], torch.Tensor] = fitness_function
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.population: torch.Tensor = self._create_initial_population()

    def _create_initial_population(self) -> torch.Tensor:
        """Generate the initial population with normally distributed vectors."""
        return torch.normal(0.0, 0.1, size=(self.population_size, self.vector_length), device=self.device)

    def _mutate(self, individuals: torch.Tensor, mutation_std: float = None) -> torch.Tensor:
        """Mutate individuals by adding normally distributed noise."""
        mut_std = mutation_std if mutation_std is not None else self.mutation_stddev
        noise = torch.normal(0.0, mut_std, size=individuals.size(), device=self.device)
        return individuals + noise

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform arithmetic crossover to generate offspring."""
        alpha = torch.rand(1, device=self.device).item()
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        return offspring1, offspring2

    def _select(self, fitnesses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select two parents using roulette wheel selection."""
        probabilities = fitnesses / torch.sum(fitnesses)
        indices = torch.multinomial(probabilities, 2, replacement=False)
        return self.population[indices[0]], self.population[indices[1]]

    def run(self) -> Tuple[torch.Tensor, float]:
        """Run the Genetic Algorithm."""
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = self.fitness_function(self.population)

            # Print the best fitness of the current generation
            print(f"Generation {generation}: Best fitness = {fitnesses.max().item():.4f}")

            # Elitism: Carry over the top individuals
            elite_count = max(1, self.population_size // 10)
            elite_indices = torch.topk(fitnesses, elite_count).indices
            elite_population = self.population[elite_indices]

            # Create the next generation
            new_population = elite_population.tolist()
            dynamic_mutation_std = self.mutation_stddev * (1 - generation / self.generations)

            while len(new_population) < self.population_size:
                parent1, parent2 = self._select(fitnesses)
                offspring1, offspring2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(offspring1, dynamic_mutation_std))
                new_population.append(self._mutate(offspring2, dynamic_mutation_std))

            new_population_tensor = torch.tensor(new_population)
            self.population = new_population_tensor

        # Final result
        fitnesses = self.fitness_function(self.population)
        best_index = torch.argmax(fitnesses)
        best_solution = self.population[best_index].unsqueeze(0)

        return best_solution, fitnesses.max().item()


# Example usage
if __name__ == "__main__":
    def custom_fitness_function(population: torch.Tensor) -> torch.Tensor:
        """Example fitness function: Negative sum of squared differences from a random target."""
        target = torch.zeros(population.size(1), device=population.device)
        return -torch.sum((population - target) ** 2, dim=1)


    ga = GeneticAlgorithm(
        vector_length=40,
        population_size=100,
        generations=5000,
        mutation_stddev=0.01,
        fitness_function=custom_fitness_function,
    )
    best_solution, best_fitness = ga.run()
    print(f"Best solution found: Fitness = {best_fitness:.4f}")
    print(f"Best solution: {best_solution.cpu().numpy()}")
