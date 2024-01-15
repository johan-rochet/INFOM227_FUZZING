from fuzzingbook.Grammars import Grammar, Expansion, srange

from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.solver import ISLaSolver, SemanticError
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.MutationFuzzer import FunctionCoverageRunner
from fuzzingbook.GreyboxGrammarFuzzer import (
    FragmentMutator,
    PowerSchedule,
    LangFuzzer,
    SeedWithStructure,
)
from fuzzingbook.Coverage import population_coverage
from fuzzingbook.GrammarCoverageFuzzer import (
    GrammarCoverageFuzzer,
    extend_grammar,
    duplicate_context,
)
from fuzzingbook.GreyboxFuzzer import Seed
from fuzzingbook.Timeout import Timeout
import time
from typing import List
import string
import csv
from pandas.errors import ParserError, EmptyDataError
import pandas as pd

from tqdm import tqdm

from ordered_set import OrderedSet


# # Grammar
print("#####################")
print("CSV Grammar")
print("#####################")

list_ascii_printable = list(string.printable)
list_ascii_printable.remove('"')
list_ascii_printable.remove("\r")
list_ascii_printable.remove("\n")
list_ascii_printable.remove(",")

list_char: List[Expansion] = srange("".join(list_ascii_printable))

CSV_GRAMMMAR: Grammar = {
    "<start>": ["<csv-file>"],
    "<csv-file>": ["<hdr>", "<rows>"],
    "<rows>": ["<row>", "<row><crlf><rows>", "<row><crlf>"],
    "<hdr>": ["<row>"],
    "<row>": ["<fields>"],
    "<fields>": ["<field>", "<field><comma><fields>"],
    "<field>": ["<TEXT>", "<STRING>", ""],
    "<TEXT>": ["<character>", "<character><TEXT>"],
    "<STRING>": ["<dblquote><list_character><dblquote>", "<dblquote><dblquote>"],
    "<list_character>": [
        "<character>",
        "<character><list_character>",
        "<dblquote><dblquote><list_character>",
    ],
    "<character>": list_char,
    "<dblquote>": [chr(34)],
    "<comma>": [","],
    "<crlf>": ["\r\n"],
}
START_SYMBOL = "<start>"

dup_CSV_GRAMMMAR = extend_grammar(CSV_GRAMMMAR)

duplicate_context(dup_CSV_GRAMMMAR, "<rows>", "<row><rows>")
duplicate_context(dup_CSV_GRAMMMAR, "<rows>", "<row><crlf><rows>")
duplicate_context(dup_CSV_GRAMMMAR, "<fields>", "<field><comma><fields>")
duplicate_context(dup_CSV_GRAMMMAR, "<TEXT>", "<character><TEXT>")
duplicate_context(dup_CSV_GRAMMMAR, "<list_character>", "<character><list_character>")
duplicate_context(dup_CSV_GRAMMMAR, "<STRING>", "<dblquote><dblquote><list_character>")

dup_gram_cov_fuzzer: GrammarCoverageFuzzer = GrammarCoverageFuzzer(
    dup_CSV_GRAMMMAR, start_symbol=START_SYMBOL, max_nonterminals=50
)

solver = ISLaSolver(
    CSV_GRAMMMAR,  # type: ignore
    """    
                    exists int nb_comma :
                        exists <row> r : 
                            (count(r, "<comma>", nb_comma)
                            and 
                            forall <row> row in <rows>:
                                count(row, "<comma>", nb_comma))
                    """,
)

# # Step 1: generate valid seeds
print("#####################")
print("Step 1 : generate valid seeds")
print("#####################")


seeds: list[SeedWithStructure] = []
syntax_erroneous_inputs = []
semantic_erroneous_inputs = []


def parse_input(input: str) -> str:
    f = open("test.csv", "w")
    f.write(input)
    f.close()
    with open("test.csv", newline="") as f:
        csv.reader(f)

    pd.read_csv("test.csv", delimiter=",", engine="python")

    return input


def fuzz_for_seeds(fuzzer: GrammarFuzzer):
    fuzz = fuzzer.fuzz()

    try:
        solver.parse(fuzz, silent=True)
    except SyntaxError as e:
        syntax_erroneous_inputs.append(fuzz)

    except SemanticError as e:
        semantic_erroneous_inputs.append(fuzz)
    else:
        seeds.append(SeedWithStructure(fuzz))


seeds: list[SeedWithStructure] = []
open("tests.log", "w").close()
total_coverage = len(dup_gram_cov_fuzzer.max_expansion_coverage())
coverage = 0
for i in tqdm(range(total_coverage)):
    while coverage == i:
        fuzz_for_seeds(dup_gram_cov_fuzzer)
        coverage = total_coverage - len(
            dup_gram_cov_fuzzer.max_expansion_coverage()
            - dup_gram_cov_fuzzer.expansion_coverage()
        )


print("%d valid seeds created" % len(seeds))


print("#####################")
print("Step 2 : Generate fuzz")
print("#####################")


n = 1000
runner = FunctionCoverageRunner(parse_input)
parser = EarleyParser(CSV_GRAMMMAR)
mutator = FragmentMutator(parser)
schedule = PowerSchedule()


lang_fuzzer = LangFuzzer([seed.data for seed in seeds], mutator, schedule)

start = time.time()
lang_fuzzer.runs(runner, trials=n)
end = time.time()


# Ordered set to avoid duplicates for later performance
syntax_error = OrderedSet([])
semantic_error = OrderedSet([])
other_error = OrderedSet([])
parsed_inputs = OrderedSet([])


def sort_seed(seed: Seed) -> int:
    try:
        solver.parse(seed.data, silent=True)

    except SyntaxError:
        syntax_error.add(seed.data)
        return 0
    except SemanticError:
        semantic_error.add(seed.data)
        return 0
    except Exception:
        other_error.add(seed.data)
        return 0
    else:
        parsed_inputs.add(seed.data)
        return 1


coverage, _ = population_coverage(lang_fuzzer.inputs, parse_input)

for seed in lang_fuzzer.inputs:
    # reuse memoized information
    if hasattr(seed, "has_structure"):
        sort_seed(seed)  # type: ignore
    else:
        if isinstance(seed, str):
            seed = Seed(seed)
        sort_seed(seed)

total_inputs = (
    len(parsed_inputs) + len(syntax_error) + len(semantic_error) + len(other_error)
)

print(
    "From the %d different generated inputs, %d (%0.2f%%) can be parsed.\n"
    "In total, %d statements are covered."
    % (
        total_inputs,
        len(parsed_inputs),
        100 * len(parsed_inputs) / total_inputs,
        len(coverage),
    )
)

print("The lang fuzzer generated :")
print("-----------------------------")
print("# Correct inputs: %d" % len(parsed_inputs))

print("# Invalid syntax inputs: %d" % len(syntax_error))

print("# Invalid semantic inputs: %d" % len(semantic_error))

print("# Other errors inputs: %d" % len(other_error))
print("-----------------------------")


print("#####################")
print("Step 3 : Check inputs on the target function")
print("#####################")


wrongly_parsed_inputs = []
wrongly_parsed_inputs_syntax = []
wrongly_parsed_inputs_semantic = []
wrongly_parsed_inputs_other = []


def check_correct_input(input: str) -> None:
    try:
        parse_input(input)
    except Exception as e:
        wrongly_parsed_inputs.append(input)


def check_semantic_incorrect_input(input: str) -> None:
    try:
        parse_input(input)
        wrongly_parsed_inputs_semantic.append(input)

    except (ParserError, EmptyDataError):
        pass
    except Exception as e:
        wrongly_parsed_inputs_semantic.append(input)


def check_syntax_incorrect_input(input: str) -> None:
    try:
        parse_input(input)
        wrongly_parsed_inputs_syntax.append(input)

    except (ParserError, EmptyDataError):
        pass
    except Exception as e:
        wrongly_parsed_inputs_other.append(input)


def check_other_incorrect_input(input: str) -> None:
    try:
        parse_input(input)
        wrongly_parsed_inputs_other.append(input)

    except Exception as e:
        pass


for correct_input in parsed_inputs:
    check_correct_input(correct_input)

for syntax_error_input in syntax_error:
    check_syntax_incorrect_input(syntax_error_input)

for semantic_error_input in semantic_error:
    check_semantic_incorrect_input(semantic_error_input)

for other_error_input in other_error:
    check_other_incorrect_input(other_error_input)


print("Correctly handled inputs :")
print("-----------------------------")
print("# wrongly handled inputs: %d" % len(wrongly_parsed_inputs))
print(wrongly_parsed_inputs)
print("-------")
print(
    "# wrongly handled syntaxically wrong inputs: %d"
    % len(wrongly_parsed_inputs_syntax)
)
print(wrongly_parsed_inputs_syntax)
print("-------")
print(
    "# wrongly handled semantically wrong inputs: %d"
    % len(wrongly_parsed_inputs_semantic)
)
print(wrongly_parsed_inputs_semantic)
print("-------")
print("# wrongly handled other wrong inputs: %d" % len(wrongly_parsed_inputs_other))
print(wrongly_parsed_inputs_other)
print("-----------------------------")
