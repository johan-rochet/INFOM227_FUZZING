from fuzzingbook.Grammars import Grammar, Expansion, srange

from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.solver import ISLaSolver, SemanticError
from fuzzingbook.Parser import EarleyParser, display_tree
from fuzzingbook.MutationFuzzer import FunctionCoverageRunner
from fuzzingbook.GreyboxGrammarFuzzer import (
    FragmentMutator,
    PowerSchedule,
    LangFuzzer,
    SeedWithStructure,
    print_stats,
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
import logging
from pandas.errors import ParserError, EmptyDataError
import pandas as pd

from tqdm import tqdm

from ordered_set import OrderedSet


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
    error = False
    fuzz = fuzzer.fuzz()

    try:
        solver.parse(fuzz)
    except SyntaxError as e:
        syntax_erroneous_inputs.append(fuzz)
        error = True
    except SemanticError as e:
        semantic_erroneous_inputs.append(fuzz)
        error = True

    try:
        seeds.append(SeedWithStructure(parse_input(fuzz)))
    except ParserError as e:
        if not error:
            logging.error("ParserError : " + str(e) + "\n" + fuzz + "\n")

    except EmptyDataError as e:
        if not error:
            logging.error("EmptyDataError : " + str(e) + "\n" + fuzz + "\n")


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

has_structure = 0
for seed in lang_fuzzer.inputs:
    # reuse memoized information
    if hasattr(seed, "has_structure"):
        has_structure += sort_seed(seed)  # type: ignore
    else:
        if isinstance(seed, str):
            seed = Seed(seed)
        has_structure += sort_seed(seed)


print(
    "From the %d generated inputs, %d (%0.2f%%) can be parsed.\n"
    "In total, %d statements are covered."
    % (
        len(lang_fuzzer.inputs),
        has_structure,
        100 * has_structure / len(lang_fuzzer.inputs),
        len(coverage),
    )
)

print("The lang fuzzer generated :")
print("Correct input number: ")
print(len(parsed_inputs))
print("Syntax error number: ")
print(len(syntax_error))
print("Semantic error number: ")
print(len(semantic_error))
print("Other error number: ")
print(len(other_error))


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


print("Good inputs but rejected by the parser:")
print(len(wrongly_parsed_inputs))
print(wrongly_parsed_inputs)
print("Syntax errors not detected by the parser:")
print(len(wrongly_parsed_inputs_syntax))
print(wrongly_parsed_inputs_syntax)
print("Semantic errors not detected by the parser:")
print(len(wrongly_parsed_inputs_semantic))
print(wrongly_parsed_inputs_semantic)
print("Other errors not detected by the parser:")
print(wrongly_parsed_inputs_other)
print(len(wrongly_parsed_inputs_other))
