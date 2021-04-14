class Individual:
    def __init__(self, x_phenotype: float, y_phenotype: float, genotype_representation: str = '16-bit'):
        """
        Create individual with two phenotype features which can be converted to x_genotype and y_genotype.
        It also stores adaptation function value.
        :param x_phenotype:
        :param y_phenotype:
        :param genotype_representation: One of: '8-bit', '16-bit' - defies length of genotype
        """
        self.x_phenotype = x_phenotype
        self.y_phenotype = y_phenotype
        self.genotype_representation = genotype_representation
        self.x_genotype = []
        self.y_genotype = []
        self.adaptation_value = None
        self._init_x_genotype()
        self._init_y_genotype()

    def _init_x_genotype(self):
        self.transform_x_phenotype_to_genotype(self.x_phenotype)

    def _init_y_genotype(self):
        self.transform_y_phenotype_to_genotype(self.y_phenotype)

    def set_adaptation_value(self, value):
        self.adaptation_value = value

    def get_adaptation_value(self):
        return self.adaptation_value

    def transform_x_genotype_to_phenotype(self, x_genotype):
        genotype_representation = self.genotype_representation
        x_phenotype = 0

        if genotype_representation == '8-bit':
            exponents_of_two = [2, 1, 0, -1, -2, -3, -4]
        elif genotype_representation == '16-bit':
            exponents_of_two = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
        else:
            raise ValueError("Representation not supported. Try '8-bit' or '16-bit'.")

        for exponent in range(len(exponents_of_two)):
            x_phenotype += int(x_genotype[exponent+1]) * pow(2, exponents_of_two[exponent])

        if x_genotype[0] == 1:
            self.x_phenotype = - x_phenotype
        else:
            self.x_phenotype = x_phenotype

    def transform_y_genotype_to_phenotype(self, y_genotype):
        genotype_representation = self.genotype_representation
        y_phenotype = 0

        if genotype_representation == '8-bit':
            exponents_of_two = [2, 1, 0, -1, -2, -3, -4]
        elif genotype_representation == '16-bit':
            exponents_of_two = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
        else:
            raise ValueError("Representation not supported. Try '8-bit' or '16-bit'.")

        for exponent in range(len(exponents_of_two)):
            y_phenotype += int(y_genotype[exponent+1]) * pow(2, exponents_of_two[exponent])

        if y_genotype[0] == 1:
            self.y_phenotype = - y_phenotype
        else:
            self.y_phenotype = y_phenotype

    def transform_x_phenotype_to_genotype(self, x_phenotype):
        genotype_representation = self.genotype_representation
        x_genotype = []
        value_to_transform = x_phenotype

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
            x_genotype.append(1)
            value_to_transform = abs(value_to_transform)
        else:
            x_genotype.append(0)

        # Exponent and mantissa
        for exponent in exponents_of_two:
            if value_to_transform / pow(2, exponent) >= 1:
                x_genotype.append(1)
                value_to_transform -= pow(2, exponent)
            else:
                x_genotype.append(0)

        self.x_genotype = x_genotype

    def transform_y_phenotype_to_genotype(self, y_phenotype):
        genotype_representation = self.genotype_representation
        y_genotype = []
        value_to_transform = y_phenotype

        if genotype_representation == '8-bit':
            if value_to_transform > 8:
                raise ValueError("Too large number. Expected y_phenotype < 8")
            else:
                exponents_of_two = [2, 1, 0, -1, -2, -3, -4]

        elif genotype_representation == '16-bit':
            if value_to_transform > 64:
                raise ValueError("Too large number. Expected y_phenotype < 64")
            else:
                exponents_of_two = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

        else:
            raise ValueError("Representation not supported. Try '8-bit' or '16-bit'.")

        # Sign
        if value_to_transform < 0:
            y_genotype.append(1)
            value_to_transform = abs(value_to_transform)
        else:
            y_genotype.append(0)

        # Exponent and mantissa
        for exponent in exponents_of_two:
            if abs(value_to_transform) / pow(2, exponent) >= 1:
                y_genotype.append(1)
                value_to_transform -= pow(2, exponent)
            else:
                y_genotype.append(0)

        self.y_genotype = y_genotype
