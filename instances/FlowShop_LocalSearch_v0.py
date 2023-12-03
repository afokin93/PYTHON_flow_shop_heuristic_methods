from typing import TextIO
import sys
from itertools import product


class Componentes:
    def __init__(self, job_id, processing_times):
        #self.problem = problem
        self.job_id = job_id
        self.processing_times = processing_times

    def __str__(self):
        return f"{self.job_id}"

    def __repr__(self):
        return self.__str__()

    def get_processing_times(self):
        return self.problem.processing_times[self.job_id[0]]


class FlowShopProblem:
    def __init__(self, processing_times):
        self.processing_times = processing_times
        self.num_jobs = len(processing_times)
        self.num_machines = len(processing_times[0])

    def calculate_makespan(self, sequence):
        completion_times = [[0 for _ in range(self.num_machines)] for _ in range(self.num_jobs)]
        for job_index in range(self.num_jobs):
            # Processamento do primeiro elemento do job
            for machine in range(self.num_machines):
                completion_times[job_index][machine] = max(completion_times[job_index - 1][machine],
                                                           completion_times[job_index][machine - 1]) + sequence[job_index].processing_times[machine]
        makespan = completion_times[-1][-1]
        return makespan


class FlowShopSolution:
    def __init__(self, problem, sequence):
        self.problem = problem
        self.sequence = sequence
        self.makespan = self.calculate_makespan()

    def calculate_makespan(self):
        return self.problem.calculate_makespan(self.sequence)

    def generate_swap_moves(self):
        swap_moves = []
        swap_moves.append('null')
        for i in range(len(self.sequence)):
            for j in range(len(self.sequence)-1):
                move_pairs = (0,j+1)
                swap_moves.append(move_pairs)
        return swap_moves

    def list_of_sequences(self):
        original_sequence = [job for job in self.sequence]
        list_of_sequences = []
        while self.sequence != original_sequence or not list_of_sequences:
            for i in range(len(self.sequence) - 1):
                self.sequence[i], self.sequence[i + 1] = self.sequence[i + 1], self.sequence[i]
                list_of_sequences.append([job for job in self.sequence])
        list_of_sequences.insert(0, original_sequence)
        return list_of_sequences
    
    
class DataProcessor:
    def __init__(self, file_object: TextIO = sys.stdin):
        self.file_object = file_object
        self.data = None
        self.transposed_data = None
        self.num_jobs = None
        self.num_machines = None
        self.problem = None  # Adicionamos uma instância de FlowShopProblem

    def process_and_print_data(self):
        # Read data from the file
        self.data = self.file_object.read()
        # Dividir os dados em linhas
        lines = self.data.strip().split("\n")
        # Remover a primeira linha
        lines.pop(0)
        # Converter cada linha para uma lista de inteiros
        rows_as_int = [[int(value) for value in line.split()] for line in lines]
        # Transpor os dados
        self.transposed_data = list(map(list, zip(*rows_as_int)))
        # Definir num_jobs e num_machines
        self.num_jobs = len(self.transposed_data)
        self.num_machines = len(self.transposed_data[0])
        # Criar instância da classe Componentes com base nos dados transpostos
        componentes_list = [Componentes(i+1, self.transposed_data[i]) for i in range(self.num_jobs)]
        # Imprimir resultados
        print("num_jobs:", self.num_jobs)
        print("num_machines:", self.num_machines)
        print("\nJobs List:")
        for i, row in enumerate(self.transposed_data):
            print(f"Job: {i+1}, {row}")
        print("\nMachine List:")
        # Imprimir cada linha como uma lista de elementos
        for i, line in enumerate(lines):
            print(f"Machine: {i+1}, {list(map(int, line.split()))}")

        # Criar instância da classe FlowShopProblem com base nos dados transpostos
        self.problem = FlowShopProblem(self.transposed_data)

        # Imprimir resultados
        print("\nJob sequence:")
        print(componentes_list)

        # Calcular e imprimir o makespan
        sequence = [Componentes(i+1, self.transposed_data[i]) for i in range(self.num_jobs)]
        makespan = self.problem.calculate_makespan(sequence)
        print("\nMakespan:", makespan)

        # Criar instância da classe FlowShopSolution
        flowshop_solution = FlowShopSolution(self.problem, componentes_list)
        # Calcular e imprimir a lista de movimentos de troca
        swap_moves = flowshop_solution.generate_swap_moves()
        print("\nSwap moves:", swap_moves)
        # Calcular e imprimir a lista de sequências possíveis
        list_of_sequences = flowshop_solution.list_of_sequences()
        print("\nList of sequences:", list_of_sequences, "\n")
        # Mostrar movimentos e sequencias resultantes dos movimentos
        for i in range(len(swap_moves)):
            makespan = flowshop_solution.problem.calculate_makespan(list_of_sequences[i])
            if i == 0:
                print(f"No move: {swap_moves[i]} --> {list_of_sequences[i]}, Makespan: {makespan}")
            else:        
                print(f"Move {i}: {swap_moves[i]} --> {list_of_sequences[i]}, Makespan: {makespan}")


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância da classe DataProcessor
    processor = DataProcessor()
    # Ler dados do arquivo (stdin)
    processor.process_and_print_data()
