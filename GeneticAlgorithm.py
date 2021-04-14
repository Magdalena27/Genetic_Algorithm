from Individual import Individual
import random
from math import sqrt
import numpy as np
import copy


class GeneticAlgorithm:
    def __init__(self, adaptation_function, number_of_individuals, ax, bx, ay, by, selection_type, n_best_individuals,
                 pk, k,
                 order_established, pm, m, stop_condition, iterations=1000, stop_function_value=100,
                 percent_of_bests=0.5,
                 genotype_representation='16-bit'):
        """
        :param adaptation_function:
        :param number_of_individuals: int
        :param ax, bx defines range of possible x_phenotype values
        :param ay, by defines range of possible y_phenotype values
        :param selection_type: str 'ranking' or 'roulette_wheel'
        :param n_best_individuals: int n_best_individuals < number_of_individuals/2 - used with 'ranking' model
        :param pk: float, crossing-over probability
        :param k: int, number of crossing-over points
        :param order_established: bool, if True phenotype fragments will be assigned to children alternately,
                                        if False - randomly
        :param pm: float, mutation probability
        :param m: mutation points
        :param stop_condition: str 'number_of_iterations' or 'function_value' or 'percent_of_bests'
        :param iterations: int, number of iterations of algorithm, used with 'number_of_iterations' stop condition
        :param stop_function_value: float, target value function, used with 'function_value' or 'percent_of_bests' stop condition
        :param percent_of_bests: float from range (0, 1], used with 'percent_of_bests' stop condition
        :param genotype_representation: '8-bit' or '16-bit'
        """

        self.by = by
        self.bx = bx
        self.ay = ay
        self.ax = ax
        self.genotype_representation = genotype_representation
        self.percent_of_bests = percent_of_bests
        self.stop_function_value = stop_function_value
        self.iterations = iterations
        self.stop_condition = stop_condition
        self.m = m
        self.pm = pm
        self.order_established = order_established
        self.k = k
        self.pk = pk
        self.n_best_individuals = n_best_individuals
        self.adaptation_function = adaptation_function
        self.number_of_individuals = number_of_individuals
        self.selection_type = selection_type
        self.population = []
        self._init_population()
        if self.genotype_representation == '8-bit':
            self.bits = 8
        elif self.genotype_representation == '16-bit':
            self.bits = 16
        else:
            raise ValueError("Genotype representation not supported. Try '8-bit' or '16-bit'")

    def _init_population(self):
        for i in range(self.number_of_individuals):
            x_phenotype = random.randint(self.ax, self.bx)
            y_phenotype = random.randint(self.ay, self.by)
            individual = Individual(x_phenotype, y_phenotype, self.genotype_representation)
            self.population.append(individual)

    def evaluate_individuals(self, f):
        for individual in self.population:
            individual.transform_x_genotype_to_phenotype(individual.x_genotype)
            individual.transform_y_genotype_to_phenotype(individual.y_genotype)
            x_phenotype = individual.x_phenotype
            y_phenotype = individual.y_phenotype
            individual.set_adaptation_value(f(x_phenotype, y_phenotype))

    def select_next_generation(self, selection_type):
        population = self.population

        if selection_type == 'ranking':
            if self.n_best_individuals > self.number_of_individuals / 2:
                raise ValueError("n_best_individuals too high!")
            else:
                sorted_population = sorted(population, key=lambda individual: individual.adaptation_value, reverse=True)
                sorted_population = sorted_population[: (len(population) - self.n_best_individuals)]
                sorted_population.extend(sorted_population[: self.n_best_individuals])
                return sorted_population
        elif selection_type == 'roulette_wheel':
            probability_of_survive = []
            sum_of_probability = 0
            adaptation_values = []
            for individual in population:
                adaptation_values.append(individual.get_adaptation_value())
            min_val_to_0 = abs(min(adaptation_values))
            for individual in population:
                sum_of_probability += sqrt(individual.get_adaptation_value() + min_val_to_0)
            for individual in population:
                probability = sqrt(individual.get_adaptation_value() + min_val_to_0) / sum_of_probability
                probability_of_survive.append(probability)
            new_population = np.random.choice(population, self.number_of_individuals, p=probability_of_survive)
            return list(new_population)

    def mix_population(self):
        random.shuffle(self.population)

    def transform_binary_to_decimal(self, value):
        genotype_representation = self.genotype_representation
        decimal_number = 0

        if genotype_representation == '8-bit':
            exponents_of_two = [2, 1, 0, -1, -2, -3, -4]
        elif genotype_representation == '16-bit':
            exponents_of_two = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
        else:
            raise ValueError("Representation not supported. Try '8-bit' or '16-bit'.")

        for exponent in range(len(exponents_of_two)):
            decimal_number += int(value[exponent+1]) * pow(2, exponents_of_two[exponent])

        if value[0] == 1:
            return -decimal_number
        else:
            return decimal_number

    def transform_decimal_to_binary(self, value):
        genotype_representation = self.genotype_representation
        binary_number = []
        value_to_transform = value

        if genotype_representation == '8-bit':
            if value_to_transform > 8:
                raise ValueError("Too large number. Expected x_phenotype < 8")
            else:
                exponents_of_two = [2, 1, 0, -1, -2, -3, -4]

        elif genotype_representation == '16-bit':
            if value_to_transform > 64:
                raise ValueError("Too large number. Expected x_phenotype < 64")
            else:
                exponents_of_two = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

        else:
            raise ValueError("Representation not supported. Try '8-bit' or '16-bit'.")

        # Sign
        if value_to_transform < 0:
            binary_number.append(1)
            value_to_transform = abs(value_to_transform)
        else:
            binary_number.append(0)

        # Exponent and mantissa
        for exponent in exponents_of_two:
            if value_to_transform / pow(2, exponent) >= 1:
                binary_number.append(1)
                value_to_transform -= pow(2, exponent)
            else:
                binary_number.append(0)

        return binary_number

    def cross_over(self, pk, k, order_established):
        population = self.population
        population_pairs = round(len(population) / 2)
        new_population = []

        if order_established:
            for pair in range(population_pairs):
                random_value = random.randint(0, 1)
                if random_value < pk:
                    first_parent_x = population[0].x_genotype
                    first_parent_y = population[0].y_genotype
                    second_parent_x = population[1].x_genotype
                    second_parent_y = population[1].y_genotype
                    locuses = self.draw_locuses(k)
                    first_child_x, first_child_y, second_child_x, second_child_y = [], [], [], []
                    first_child_x.extend(first_parent_x[:locuses[0]])
                    second_child_x.extend(second_parent_x[:locuses[0]])
                    for iteration in range(1, len(locuses)):
                        if iteration % 2 == 0:
                            l1 = locuses[iteration - 1]
                            l2 = locuses[iteration]
                            first_child_x.extend(first_parent_x[l1: l2])
                            second_child_x.extend(second_parent_x[l1: l2])
                        else:
                            first_child_x.extend(second_parent_x[locuses[iteration - 1]: locuses[iteration]])
                            second_child_x.extend(first_parent_x[locuses[iteration - 1]: locuses[iteration]])

                    if k % 2 == 0:
                        first_child_x.extend(first_parent_x[locuses[k-1]:])
                        second_child_x.extend(second_parent_x[locuses[k-1]:])
                    else:
                        first_child_x.extend(second_parent_x[locuses[k - 1]:])
                        second_child_x.extend(first_parent_x[locuses[k - 1]:])

                    # Avoiding crossing borders
                    first_child_x_decimal = self.transform_binary_to_decimal(first_child_x)
                    if first_child_x_decimal <= self.ax:
                        first_child_x = self.transform_decimal_to_binary(self.ax)
                    elif first_child_x_decimal >= self.bx:
                        first_child_x = self.transform_decimal_to_binary(self.bx)

                    second_child_x_decimal = self.transform_binary_to_decimal(second_child_x)
                    if second_child_x_decimal <= self.ax:
                        second_child_x = self.transform_decimal_to_binary(self.ax)
                    elif second_child_x_decimal >= self.bx:
                        second_child_x = self.transform_decimal_to_binary(self.bx)

                    locuses = self.draw_locuses(k)
                    first_child_y.extend(first_parent_y[:locuses[0]])
                    second_child_y.extend(second_parent_y[:locuses[0]])
                    for iteration in range(1, len(locuses)):
                        if iteration % 2 == 0:
                            first_child_y.extend(first_parent_y[locuses[iteration - 1]: locuses[iteration]])
                            second_child_y.extend(second_parent_y[locuses[iteration - 1]: locuses[iteration]])
                        else:
                            first_child_y.extend(second_parent_y[locuses[iteration - 1]: locuses[iteration]])
                            second_child_y.extend(first_parent_y[locuses[iteration - 1]: locuses[iteration]])

                    if k % 2 == 0:
                        first_child_y.extend(first_parent_y[locuses[k-1]:])
                        second_child_y.extend(second_parent_y[locuses[k-1]:])
                    else:
                        first_child_y.extend(second_parent_y[locuses[k - 1]:])
                        second_child_y.extend(first_parent_y[locuses[k - 1]:])

                    # Avoiding crossing borders
                    first_child_y_decimal = self.transform_binary_to_decimal(first_child_y)
                    if first_child_y_decimal <= self.ay - 0.001:
                        first_child_y = self.transform_decimal_to_binary(self.ay)
                    elif first_child_y_decimal >= self.by:
                        first_child_y = self.transform_decimal_to_binary(self.by)

                    second_child_y_decimal = self.transform_binary_to_decimal(second_child_y)
                    if second_child_y_decimal <= self.ay - 0.001:
                        second_child_y = self.transform_decimal_to_binary(self.ay)
                    elif second_child_y_decimal >= self.by:
                        second_child_y = self.transform_decimal_to_binary(self.by)

                    new_population.append(copy.deepcopy(population[0]))
                    last_index = len(new_population) - 1
                    new_population[last_index].x_genotype = first_child_x
                    new_population[last_index].y_genotype = first_child_y
                    new_population.append(copy.deepcopy(population[1]))
                    last_index += 1
                    new_population[last_index].x_genotype = second_child_x
                    new_population[last_index].y_genotype = second_child_y
                    population.remove(population[1])
                    population.remove(population[0])
                else:
                    new_population.append(copy.deepcopy(population[0]))
                    new_population.append(copy.deepcopy(population[1]))
                    population.remove(population[1])
                    population.remove(population[0])

            if len(population) == 1:
                new_population.append(population[0])

            self.population = new_population

        elif not order_established:
            for pair in range(population_pairs):
                random_value = random.randint(0, 1)
                if random_value < pk:
                    first_parent_x = population[0].x_genotype
                    first_parent_y = population[0].y_genotype
                    second_parent_x = population[1].x_genotype
                    second_parent_y = population[1].y_genotype

                    locuses = self.draw_locuses(k)
                    first_child_x, first_child_y, second_child_x, second_child_y = [], [], [], []
                    order_for_first_child = random.randint(0, 1, k + 1)
                    order_for_second_child = random.randint(0, 1, k + 1)
                    orders = [order_for_first_child, order_for_second_child]
                    first_child_x.extend([first_parent_x, second_parent_x][order_for_first_child[0]][:locuses[0]])
                    second_child_x.extend([first_parent_x, second_parent_x][order_for_second_child[0]][:locuses[0]])
                    children_x = [first_child_x, second_child_x]

                    for iteration in range(1, len(locuses)):
                        for i in range(2):
                            order = orders[i]
                            if order == 0:
                                children_x[i].extend(first_parent_x[locuses[iteration - 1]: locuses[iteration]])
                            else:
                                children_x[i].extend(second_parent_x[locuses[iteration - 1]: locuses[iteration]])
                    first_child_x.extend([first_parent_x, second_parent_x][order_for_first_child[-1]][locuses[-1]:])
                    second_child_x.extend([first_parent_x, second_parent_x][order_for_second_child[-1]][locuses[-1]:])

                    # Avoiding crossing borders
                    first_child_x_decimal = self.transform_binary_to_decimal(first_child_x)
                    if first_child_x_decimal <= self.ax:
                        first_child_x = self.transform_decimal_to_binary(self.ax)
                    elif first_child_x_decimal >= self.bx:
                        first_child_x = self.transform_decimal_to_binary(self.bx)

                    second_child_x_decimal = self.transform_binary_to_decimal(second_child_x)
                    if second_child_x_decimal <= self.ax:
                        second_child_x = self.transform_decimal_to_binary(self.ax)
                    elif second_child_x_decimal >= self.bx:
                        second_child_x = self.transform_decimal_to_binary(self.bx)

                    locuses = self.draw_locuses(k)
                    order_for_first_child = random.randint(0, 1, k + 1)
                    order_for_second_child = random.randint(0, 1, k + 1)
                    orders = [order_for_first_child, order_for_second_child]
                    first_child_y.extend([first_parent_y, second_parent_y][order_for_first_child[0]][:locuses[0]])
                    second_child_y.extend([first_parent_y, second_parent_y][order_for_second_child[0]][:locuses[0]])
                    children_y = [first_child_y, second_child_y]

                    for iteration in range(1, len(locuses)):
                        for i in range(2):
                            order = orders[i]
                            if order == 0:
                                children_y[i].extend(first_parent_y[locuses[iteration - 1]: locuses[iteration]])
                            else:
                                children_y[i].extend(second_parent_y[locuses[iteration - 1]: locuses[iteration]])
                    first_child_y.extend([first_parent_y, second_parent_y][order_for_first_child[-1]][locuses[-1]:])
                    second_child_y.extend([first_parent_y, second_parent_y][order_for_second_child[-1]][locuses[-1]:])

                    # Avoiding crossing borders
                    first_child_y_decimal = self.transform_binary_to_decimal(first_child_y)
                    if first_child_y_decimal <= self.ay - 0.001:
                        first_child_y = self.transform_decimal_to_binary(self.ay)
                    elif first_child_y_decimal >= self.by:
                        first_child_y = self.transform_decimal_to_binary(self.by)

                    second_child_y_decimal = self.transform_binary_to_decimal(second_child_y)
                    if second_child_y_decimal <= self.ay - 0.001:
                        second_child_y = self.transform_decimal_to_binary(self.ay)
                    elif second_child_y_decimal >= self.by:
                        second_child_y = self.transform_decimal_to_binary(self.by)

                    new_population.append(copy.deepcopy(population[0]))
                    last_index = len(new_population)
                    new_population[last_index].x_genotype = first_child_x
                    new_population[last_index].y_genotype = first_child_y
                    new_population.append(copy.deepcopy(population[1]))
                    last_index += 1
                    new_population[last_index].x_genotype = second_child_x
                    new_population[last_index].y_genotype = second_child_y

                    population.remove([population[0], [1]])

                else:
                    new_population.append(copy.deepcopy(population[0]))
                    new_population.append(copy.deepcopy(population[1]))
                    population.remove([population[0], population[1]])

            if len(population) == 1:
                new_population.append(population[0])

            self.population = new_population

    def draw_locuses(self, k):
        locuses = []
        while len(locuses) < k:
            locus = np.random.randint(1, self.bits - 1)
            if locus not in locuses:
                locuses.append(locus)
        locuses.sort()

        return locuses

    def mutate_individuals(self, pm, m):
        population = self.population

        for individual in population:
            # for x feature
            random_value = random.randint(0, 1)
            if random_value < pm:
                locuses_to_mutate = self.draw_mutation_locuses(m, self.ax, self.bx)
                for locus in locuses_to_mutate:
                    if individual.x_genotype[locus] == '0':
                        individual.x_genotype[locus] = '1'
                    else:
                        individual.x_genotype[locus] = '0'

                # Avoiding crossing borders
                individual.transform_x_genotype_to_phenotype(individual.x_genotype)
                if individual.x_phenotype <= self.ax + 0.001:
                    individual.x_phenotype = self.ax
                elif individual.x_phenotype >= self.bx - 0.001:
                    individual.x_phenotype = self.bx
            else:
                pass

            # for y feature
            random_value = random.randint(0, 1)
            if random_value < pm:
                locuses_to_mutate = self.draw_mutation_locuses(m, self.ay, self.by)
                for locus in locuses_to_mutate:
                    if individual.y_genotype[locus] == '0':
                        individual.y_genotype[locus] = '1'
                    else:
                        individual.y_genotype[locus] = '0'

                # Avoiding crossing borders
                individual.transform_y_genotype_to_phenotype(individual.y_genotype)
                if individual.y_phenotype <= self.ay + 0.001:
                    individual.y_phenotype = self.ay
                elif individual.y_phenotype >= self.by - 0.001:
                    individual.y_phenotype = self.by
            else:
                pass

        return population

    def draw_mutation_locuses(self, m, a, b):
        locuses = []
        mutation_border = min(abs(a), abs(b))
        binary_mutation_border = self.to_binary_border(mutation_border)
        while len(locuses) < m:
            locus = np.random.randint(max(0, binary_mutation_border), self.bits)
            if locus not in locuses:
                locuses.append(locus)
        locuses.sort()

        return locuses

    def to_binary_border(self, border):
        binary_border = self.transform_decimal_to_binary(border)

        for i in range(1, len(binary_border)):
            if binary_border[i] == 1:
                return i+1
        return 0

    def perform_calculation(self):
        stop_condition = self.stop_condition

        if stop_condition == 'number_of_iterations':
            iters = self.iterations
            for i in range(iters):
                self.evaluate_individuals(self.adaptation_function)
                self.population = self.select_next_generation(self.selection_type)
                self.cross_over(self.pk, self.k, self.order_established)
                self.population = self.mutate_individuals(self.pm, self.m)

        elif stop_condition == 'function_value':
            stop_function_value = self.stop_function_value
            actual_function_value = 0
            while actual_function_value < stop_function_value:
                self.evaluate_individuals(self.adaptation_function)
                individuals_function_value = []
                for individual in self.population:
                    individuals_function_value.append(individual.get_adaptation_value())
                actual_function_value = max(individuals_function_value)
                self.population = self.select_next_generation(self.selection_type)
                self.cross_over(self.pk, self.k, self.order_established)
                self.population = self.mutate_individuals(self.pm, self.m)

        elif stop_condition == 'percent_of_bests':
            target_percent_of_bests = self.percent_of_bests
            stop_function_value = self.stop_function_value
            actual_percent_of_bests = 0.0
            while actual_percent_of_bests < target_percent_of_bests:
                self.evaluate_individuals(self.adaptation_function)
                individuals_function_value = []
                for individual in self.population:
                    individuals_function_value.append(individual.get_adaptation_value())
                actual_percent_of_bests = len(
                    [i for i in individuals_function_value if i >= stop_function_value]) / self.number_of_individuals
                self.population = self.select_next_generation(self.selection_type)
                self.cross_over(self.pk, self.k, self.order_established)
                self.population = self.mutate_individuals(self.pm, self.m)

        else:
            raise ValueError(
                "Stop condition not supported. Try: 'number_of_iterations' or 'function_value' or 'percent_of_bests'.")

        individuals_function_value = []
        for individual in self.population:
            individuals_function_value.append(individual.get_adaptation_value())
        best_value = max(individuals_function_value)

        for individual in self.population:
            if individual.get_adaptation_value() == best_value:
                return individual

    def get_result(self):
        individual = self.perform_calculation()
        x_val = individual.x_phenotype
        y_val = individual.y_phenotype
        if x_val < self.ax:
            x_val = self.ax
        if x_val > self.bx:
            x_val = self.bx
        if y_val < self.ay:
            y_val = self.ay
        if y_val > self.by:
            y_val = self.by
        individual.set_adaptation_value(self.adaptation_function(x_val, y_val))
        f_val = individual.get_adaptation_value()
        return x_val, y_val, f_val