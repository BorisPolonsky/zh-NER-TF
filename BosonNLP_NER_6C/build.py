import os
import re
import sys


def get_lines(file_handle, internal_lf_process="remove"):
    """
    A generator for yielding sentences (with final lf stripped).
    :param file_handle: File handle for BosonNLP_NER_6C returned by built-in open() function.
    :param internal_lf_process: Can be specified as either ignore/remove/split.
    The original file use "\\n" notations to represent
    line feeds in case multiple lines in the source text are coupled and stored as a single line in the file.
    Setting this parameter will to either:
        (split) split these coupled line and yield them separately,
        (remove) substitute them with empty strings or,
        (ignore) leave the notations.
    :return: A single line of sentence/paragraph/article etc., with the tailing
    line feed "\n" stripped.
    """
    if internal_lf_process == "split":
        raise NotImplemented('Bug not fixed yet in case"\\n" notations appears within single entity structure')
        for line in file_handle:
            line = line.rstrip()
            for internal_line in line.split(r"\n"):
                yield internal_line
    elif internal_lf_process == "ignore":
        for line in file_handle:
            yield line.rstrip()
    elif internal_lf_process == "remove":
        for line in file_handle:
            yield line.replace(r"\n", "").rstrip()
    else:
        raise ValueError("Expected split/ignore/remove, got {}.".format(internal_lf_process))

def get_segments(lines):
    def split(string, by):
        if len(by) > 1:
            try:
                s, *by_ = by
            except ValueError:
                print(by)
                exit(-1)
            for seg in split(string, by=by[1:]):
                for inner_seg in seg.split(s):
                    yield inner_seg
        else:
            for seg in string.split(by[0]):
                yield seg

    for line in lines:
        for sentence in split(line, by=("。", "！", "；")):
            sentence = sentence.strip(" ")
            yield sentence

def main(args):
    if args.char_list is not None:
        raise NotImplemented("Not implemented yet.")
    # Warning, crappy implementation head!
    # How tokens are named in original text
    token_names_original = ("person_name", "product_name", "company_name", "time", "org_name", "location")
    # Suffices for each time of entity in IOB scheme.
    token_suffices_new = (args.per_token, args.prod_token, args.co_token, args.time_token, args.org_token, args.loc_token)
    # Argument check
    target_entities = []
    for t_old, t_new in zip(token_names_original, token_suffices_new):
        if t_new is None:
            sys.stdout.write("Ignoring entity of type {}".format(t_old))
        else:
            target_entities.append((t_old, t_new))
    if len(target_entities) == 0:
        sys.stderr.write("At least 1 type of desired entities should be specified. Got 0.\n")
        exit(-1)
    else:
        sys.stdout.write("\n".join(
            map(lambda x: "Characters within entities of type {} will be labeled with suffix {}".format(*x),
                target_entities)))
        sys.stdout.write("\nCharacters within non-entities will be labeled as {}.\n".format(args.non_entity_token))
    # Specify io paths
    if_path, of_path = os.path.normpath(args.input_file), os.path.normpath(args.output_file)
    # entity_re_utils[original_token] = (compiled_re_pattern, new_token_suffix)
    # Note on the following regex
    # Expression [ ]* filters out the possible spaces between ":" and entity.
    # The left bracket does not always comes in pairs in this corpus, thus \{* is used as substitution for \{\{.
    # Note that the corpus contains redundantly nested entites (e.g. {{location:{{location: Some location}}}}.
    entity_re_utils = {t_old: (re.compile(r"\{*%s:[  ]*(.*?[\}]*)\}\}" % t_old), t_new)
                       for t_old, t_new in zip(token_names_original, token_suffices_new)}

    with open(if_path, "r") as fi, open(of_path, "w", encoding='utf-8') as fo:
        outer_dropout_tag, inner_dropout_tag = object(), object()  # Used as a special label for discarding characters.
        for line in get_segments(get_lines(fi, args.internal_lf_process.lower())):
            line = line.rstrip("\n")
            if line == "":
                continue
            else:
                # Some of the entity representation are redundantly nested
                # (e.g. {{location:{{location: Some location}}}} check before proceeding.
                exist_nested_entity = True  # Assume existence first
                while exist_nested_entity:
                    exist_nested_entity = False
                    line_tokens = [args.non_entity_token] * len(line)
                    # Find each type of entity in a single line
                    for token_old, (compiled_re_pattern, new_token_suffix) in entity_re_utils.items():
                        # Find each position for labeling entities. This implementation assumes no overlap between entities.
                        for it in compiled_re_pattern.finditer(line):
                            print(it.group())
                            i_entity_begin, i_entity_end = it.span(1)  # Boundary of entity text.
                            i_match_begin, i_match_end = it.span(0)  # Boundary of whole entity tag.
                            if not compiled_re_pattern.search(line[i_entity_begin: i_entity_end]):  # Check nesting
                                if (new_token_suffix is not None) and (not exist_nested_entity):  # parse this entity tag
                                    line_tokens[i_entity_begin] = "B-%s" % new_token_suffix
                                    line_tokens[i_entity_begin + 1: i_entity_end] = \
                                        ["I-%s" % new_token_suffix] * (i_entity_end - i_entity_begin - 1)
                                line_tokens[i_match_begin: i_entity_begin] = [inner_dropout_tag] * (
                                            i_entity_begin - i_match_begin)
                                line_tokens[i_entity_end: i_match_end] = [inner_dropout_tag] * (
                                            i_match_end - i_entity_end)
                            else:
                                # Dropout outer structure without tagging entity
                                exist_nested_entity = True
                                line_tokens[i_match_begin: i_entity_begin] = [outer_dropout_tag] * (
                                        i_entity_begin - i_match_begin)
                                line_tokens[i_entity_end: i_match_end] = [outer_dropout_tag] * (
                                        i_match_end - i_entity_end)
                    if exist_nested_entity:  # Dropout outer structures and begin a new round
                        line = "".join([ch if token is not outer_dropout_tag else "" for ch, token in zip(line, line_tokens)])
            assert len(line) == len(line_tokens)
            for i, (ch, token) in enumerate(zip(line, line_tokens)):
                if token is not inner_dropout_tag and ch != args.sep:
                    fo.write("{}{}{}\n".format(ch, args.sep, token))
                    if ch == "{":
                        print(i)
                        print(tuple(zip(range(len(line)), line, line_tokens)))
                        exit(-1)
            fo.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label each character in the original text as"
                                                 "B-{ENTITY_TYPE_SUFFIX}, I-{ENTITY_TYPE_SUFFIX} or "
                                                 "{NON-ENTITY-TOKEN}")
    parser.add_argument("-i", "--input-file", default="./BosonNLP_NER_6C.txt",
                        help="Input file path for BosonNLP_NER.")
    parser.add_argument("-o", "--output-file", type=str, default="BosonNLP_parsed",
                        help="Output file path for parsed text.")
    parser.add_argument("--char-list", default=None, type=str,
                        help="If specified, all characters in the text will be saved to this path.")
    parser.add_argument("--per-token", type=str, default=None,
                        help='Suffix for entities of type "person_name" '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--prod-token", type=str, default=None,
                        help='Suffix for entities of type "product_name" in IOB scheme, '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--co-token", type=str, default=None,
                        help='Suffix for entities of type "company_name" in IOB scheme, '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--time-token", type=str, default=None,
                        help='Suffix for entities of type "time" in IOB scheme, '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--org-token", type=str, default=None,
                        help='Suffix for entities of type "org_name" in IOB scheme, '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--loc-token", type=str, default=None,
                        help='Suffix for entities of type "location" in IOB scheme, '
                             '(will be treated as non-entities if not specified) in IOB scheme')
    parser.add_argument("--non-entity-token", type=str, default="O",
                        help="Token for characters within non-entities and entities.")
    parser.add_argument("--sep", type=str, default="\t",
                        help="Delimiter between character and token.")
    parser.add_argument("--internal-lf-process", type=str, default="remove",
                        help='Can be specified as either ignore/remove/split.\n'
                             'The original file use "\\n" notations to represent '
                             'line feeds in case multiple lines in the source text were coupled as single one. '
                             'Specifying this option will split these coupled line and treat them as individual'
                             'sentences.')
    args = parser.parse_args()
    main(args)
