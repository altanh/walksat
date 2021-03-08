import sys
import glob
import os
import pathlib
import numpy as np


OUTPUT_DIR = "bench"
TRACE_DIR = "data/traces"


def run_walksat(walksat_path, cnf_path):
    problem_name = os.path.splitext(os.path.split(cnf_path)[1])[0]
    trace_file = TRACE_DIR + "/" + problem_name + ".txt"
    os.system(f"cat {cnf_path} | {walksat_path} > {trace_file}")
    return trace_file


def parse_trace(trace_path):
    problem_name = os.path.splitext(os.path.split(trace_path)[1])[0]
    X_file = OUTPUT_DIR + "/X_" + problem_name
    Y_file = OUTPUT_DIR + "/Y_" + problem_name
    with open(trace_path, "r") as trace_file:
        state = "junk"
        X = []
        Y = []
        num_vars = None
        for line in trace_file:
            line = line.strip()

            if len(line) == 0:
                continue
            elif line.startswith("NEUROSAT INPUT"):
                assert state == "junk"
                state = "input" 
                _, _, num_vars, num_clauses = line.split(" ")
                num_vars, num_clauses = int(num_vars), int(num_clauses)
                continue
            elif line == "END":
                assert state == "input"
                state = "junk"
                X = np.array(X, dtype=np.int32).T
                continue
            elif line == "ASSIGNMENT FOUND":
                assert state == "junk"
                assert num_vars is not None
                state = "output"
                continue
            elif line == "ASSIGNMENT NOT FOUND":
                print("no solution for", trace_file)
                return

            if state == "input":
                entry = [int(c) for c in line.split(" ")]
                X.append(entry)
            elif state == "output":
                lit = int(line.split(" ")[1])
                Y.append(0 if lit < 0 else 1)
        Y = np.array(Y, dtype=np.int32)
        assert len(Y) == num_vars, f"{len(Y)} != {num_vars} in {trace_file}"
    np.save(X_file, X, allow_pickle=False)
    np.save(Y_file, Y, allow_pickle=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: gen_data.py <path-to-walksat> <cnf-directory>")
        exit(1)

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)

    walksat_path = os.path.abspath(sys.argv[1])
    cnf_directory = os.path.normpath(sys.argv[2])
    cnf_folder = os.path.split(cnf_directory)[1]

    for cnf_path in glob.glob(cnf_directory + "/*.cnf"):
        trace_path = run_walksat(walksat_path, cnf_path)
        parse_trace(trace_path)
