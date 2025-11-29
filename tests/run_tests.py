import os
import sys
import subprocess
from enum import Enum
from yaml import safe_load

def equal(x, y, eps):
    return x - eps < y and y < x + eps

class TestResult(Enum):
    Failed = "failed"
    Undefined = "undefined"
    Inited = "inited"
    Success = "success"


class Config:
    def __init__(self, raw):
        self.precision = raw['test']['precision']
        self.expected_result = raw['test']['expected_result']
        self.cpu = raw['test']['cpu']
        self.type = raw['test']['type']
        self.binary = self.get_binary()
        self.flags = self.get_flags()
        self.procs = self.get_procs()

    def get_binary(self):
        if self.cpu == "single":
            return "icpx"
        else:
            return "mpiicpx"

    def get_flags(self):
        if self.cpu == "single":
            return "-lQComputations_SINGLE_NO_PLOTS"
        else:
            return "-lQComputations_CPU_CLUSTER_NO_PLOTS"

    def get_procs(self):
        if self.type == "fast":
            return [1, 4, 8]
        elif self.type == "medium":
            return [1, 4, 8, 12]
        elif self.type == "heavy":
            return [1, 3, 4, 7, 8, 12, 15]
        else:
            return [1]


class QComputationsTest:
    name: str # test name
    path: str # path to directory with test
    config: Config
    result: TestResult

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.result = TestResult.Undefined
        with open(f'{path}/config.yaml') as stream:
            try:
                self.config = Config(safe_load(stream))
                self.result = TestResult.Inited
            except yaml.YAMLError as exception:
                print(exception)

    def is_failed(self):
        print(f'Test {self.name} has status {self.result}.')
        return self.result == TestResult.Failed

    def is_succeeded(self):
        print(f'Test {self.name} has status {self.result}.')
        return self.result == TestResult.Success

    def is_inited(self):
        print(f'Test {self.name} has status {self.result}.')
        return self.result == TestResult.Inited

    def is_undefined(self):
        print(f'Test {self.name} has status {self.result}.')
        return self.result == TestResutl.Undefined

    def validate_output(self, output):
        for symbol in output:
            if not (symbol.isdigit() or symbol == " " or symbol == "."):
                return False
        return True

    def equal_with_precision(self, expected, given):
        if not self.validate_output(expected):
            print(f'Test {self.name} has unformatted expected output.')
            return False
        if not self.validate_output(given):
            print(f'Test {self.name} has unformatted program output.')
            return False

        expected_values = list(map(float, expected.split()))
        given_values = list(map(float, given.split()))
        if len(expected_values) != len(given_values):
            print(f'Test {self.name} error: expected_values {len(expected_values)} len not equal to given_values {len(given_values)} len.')
            return False
        for i in range(len(expected_values)):
            if not equal(expected_values[i], given_values[i], float(self.config.precision)):
                print(f'Test {self.name} error: diff in value on {i} position -> {expected_values[i]} != {given_values[i]} with {self.config.precision}')
                return False
        return True

    def compile(self):
        print(f'Test {self.name} compiling...')
        completed_process = subprocess.run([
            self.config.binary,
            self.config.flags,
            f'{self.path}/test.cpp',
            '-o',
            f'{self.path}/test'
        ])
        print(f'Test {self.name} compiled with status {completed_process.returncode}.')
        return completed_process.returncode == 0

    def validate(self):
        base_args = []
        if self.config.cpu == "cluster":
            base_args.append("mpirun")
            #base_args.append("--hostfile")
            #base_args.append(f"{self.path}/hostfile")
            base_args.append("-n")
        
        for nprocs in self.config.procs:
            run_args = []
            if len(base_args) != 0:
                run_args.append(str(nprocs))
            run_args.append(f'./{self.path}/test')

            print(f'Test {self.name} start on {nprocs} processes...')
            completed_process = subprocess.run(base_args + run_args, capture_output=True, check=True, text=True)
            if not completed_process.returncode == 0:
                self.result = TestResult.Failed
                print(f'Test {self.name} run-time error: \"{completed_process.stderr}\"')
                return False
            elif not self.equal_with_precision(completed_process.stdout, self.config.expected_result):
                self.result = TestResult.Failed
                print(f'Test {self.name} unexpected result error \n    expected: \"{self.config.expected_result}\" \n    given: \"{completed_process.stdout}\"')
                return False
            else:
                self.result = TestResult.Success

            print(f'Test {self.name} on {nprocs} processes finished with {self.result} status.')
        return True

    def clear_binary(self):
        subprocess.run(["rm", f"{self.path}/test"])

    def run(self):
        # run test only if it is inited, in other cases return
        if not self.is_inited():
            return False

        if not self.compile():
            return False

        if not self.validate():
            self.clear_binary()
            return False

        self.clear_binary()
        return True


def run_tests():
    tests_path = sys.argv[1]
    for testdir in os.scandir(tests_path):
        if not testdir.is_dir():
            continue
        test = QComputationsTest(testdir.name, testdir.path)
        if not test.run():
            os._exit(1)


run_tests()
# python3 run_tests.py path/to/tests
