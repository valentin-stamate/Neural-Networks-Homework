import re

# https://regexr.com to check regex


class EquationSystemParser:
    coefficient_regex = f'([+-]?[0-9]*)'
    unknown_term_regex = f'([a-z]+)'
    end_coeff_regex = f'(=)([+-]?[0-9]+)'
    term_coeff = f'{coefficient_regex}{unknown_term_regex}'

    input_regex = f'((([+-]?([0-9]?|[1-9]([0-9])*))([a-z]+))(([+-]([0-9]?|[1-9]([0-9])*))([a-z]+))*=([+-]?([0-9]?|[1-9]([0-9])*))(\n?))+'

    term_coeff_comp = re.compile(term_coeff)
    end_coeff_regex_comp = re.compile(end_coeff_regex)
    coefficient_regex_comp = re.compile(coefficient_regex)
    unknown_term_regex_comp = re.compile(unknown_term_regex)

    input_regex_comp = re.compile(input_regex)

    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):

        with open(self.file_path) as f:
            input_text = f.read()
            input_text = input_text.replace(' ', '')

        with open(self.file_path) as f:
            input_lines = [x.replace(' ', '') for x in f.readlines()]

        if not self.check_input(input_text):
            raise Exception('Invalid input file')

        unknown_terms = {x for x in self.unknown_term_regex_comp.findall(input_text)}
        unknown_terms_temp = unknown_terms

        unknown_terms = {}
        for i, term in enumerate(unknown_terms_temp):
            unknown_terms[term] = i

        matrix = []
        result = []

        for line in input_lines:
            coeff_terms = self.term_coeff_comp.findall(line)

            matrix_line = [0] * len(unknown_terms)

            for coeff_string, term in coeff_terms:
                coeff = self.raw_coefficient_to_number(coeff_string)
                matrix_line[unknown_terms[term]] = coeff

            result_coeff = int(self.end_coeff_regex_comp.findall(line)[0][1])
            result.append([result_coeff])

            matrix.append(matrix_line)
        return matrix, result, [x for x in unknown_terms_temp]

    def check_input(self, input_text):
        match = self.input_regex_comp.match(input_text)

        if match is None:
            return False

        return match.span()[1] == len(input_text)

    @staticmethod
    def raw_coefficient_to_number(raw_coefficient):
        if raw_coefficient == '' or raw_coefficient == '+':
            return 1

        if raw_coefficient == '-':
            return -1

        return int(raw_coefficient)
